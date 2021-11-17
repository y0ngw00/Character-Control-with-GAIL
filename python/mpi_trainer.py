import time
import math
import colorsys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
from collections import namedtuple
from collections import deque

import pickle
import os

import pycomcon
import ppo
import discriminator

from torch.utils.tensorboard import SummaryWriter
from mpi_utils import get_num_procs, get_proc_rank, is_root_proc, is_root2_proc,get_root_proc, get_root2_proc, broadcast, gather, scatter, send, recv

Sample = namedtuple('Sample',('s', 'a', 'rg', 'ss1', 'vf_pred', 'log_prob'))

class Trainer(object):
	def __init__(self, env_cls, config, path):
		self.env = env_cls()
		self.path = path

		self.policy = self.create_policy(torch.device("cpu"), config['model'], config['policy'])
		self.disc = self.create_disc(torch.device("cpu"), config['discriminator_model'], config['discriminator'])
		if is_root_proc():
			self.policy_loc = self.create_policy(torch.device("cuda"), config['model'], config['policy'])
		if is_root2_proc():
			self.disc_loc = self.create_disc(torch.device("cuda"), config['discriminator_model'], config['discriminator'])

		self.num_envs = get_num_procs()

		trainer_config = config['trainer']
		self.sample_size = trainer_config['sample_size']
		self.sample_epoch = self.sample_size//self.num_envs

		self.num_sgd_iter = trainer_config['num_sgd_iter']
		self.num_disc_sgd_iter = trainer_config['num_disc_sgd_iter']

		self.sgd_minibatch_size = trainer_config['sgd_minibatch_size']
		self.disc_sgd_minibatch_size = trainer_config['disc_sgd_minibatch_size']

		self.save_iteration = trainer_config['save_iteration']
		
		self.env.reset()
		self.state = self.env.get_state()

		self.episode_buffers = []
		self.episode_buffers.append([])
		self.states_expert = self.env.get_states_AMP_expert()

		self.enable_goal = self.env.is_enable_goal()

		if is_root_proc():
			self.state_dict = {}
			self.state_dict['elapsed_time'] = 0.0
			self.state_dict['num_iterations_so_far'] = 0
			self.state_dict['num_samples_so_far'] = 0

			self.create_summary_writer(path)

	def create_policy(self, device, model_config, policy_config):
		return ppo.PPO(self.env.get_dim_state(),self.env.get_dim_state_label(), self.env.get_dim_action(), device, model_config , policy_config)

	def create_disc(self, device, model_config, disc_config):
		return discriminator.Discriminator(self.env.get_dim_state_AMP(),self.env.get_num_total_label(),self.env.get_dim_state_label(), device, model_config, disc_config)

	def create_summary_writer(self, path):
		self.writer = SummaryWriter(path)

	def step(self):
		if is_root_proc():
			self._tic()
		self.sync()
		self.generate_episodes()
		self.postprocess_episodes()
		self.gather_episodes()

		if is_root_proc():
			
			valid_samples = self.concat_samples(self.policy_episode_list)
			t_sample = self._toc()

			if valid_samples:
				self.standarize_samples()
				self.update_filter()
				self._tic()
				log_policy = self.optimize()
				log_policy['t'] = self._toc()
				log_disc = recv(source=get_root2_proc())
				
				self.print_log(t_sample, log_policy, log_disc, self.writer)
				self.save(self.path)
				self.state_dict['elapsed_time'] += t_sample + np.max([log_policy['t'],log_disc['t']])
			else:
				self.state_dict['elapsed_time'] += t_sample
		if is_root2_proc():
			valid_samples = self.concat_samples(self.disc_episode_list)
			if valid_samples:
				self.update_filter()
				self._tic()
				log = self.optimize()
				log['t'] = self._toc()
				send(log, dest=get_root_proc())
			

	def _tic(self):
		self.tic = time.time()

	def _toc(self):
		t = 0.0
		t = time.time() - self.tic
		return t

	def sample_states_expert(self, n):
		m = len(self.states_expert)
		states =  self.states_expert[np.random.randint(0, m, n)]

		# sample_states = self.compress_label(states,"disc")
		return states


	def sync(self):
		if is_root_proc():
			state_dict = self.policy_loc.state_dict()
			self.policy.load_state_dict(state_dict)

		if is_root2_proc():
			state_dict = self.disc_loc.state_dict()
			self.disc.load_state_dict(state_dict)

		self.policy = broadcast(self.policy, root=get_root_proc())
		self.disc = broadcast(self.disc, root=get_root2_proc())

	def generate_episodes(self):
		for j in range(self.sample_epoch):
			action, logprob, valuef = self.policy(self.state)
			self.env.step(action)

			ss1 = self.env.get_state_AMP()
			eoe = self.env.inspect_end_of_episode()
			rg = self.env.get_reward_goal()
			
			self.episode_buffers[-1].append(Sample(self.state, action, rg, ss1, valuef, logprob))
			self.state = self.env.get_state()
			if eoe:
				if len(self.episode_buffers[-1]) != 0:
					self.episode_buffers.append([])
				self.env.reset()

	def compress_label(self, state):
		dim_label = self.env.get_dim_state_label()
		label = state[:,-dim_label:].squeeze()

		# num = self.env.get_num_total_label()
		# label_one_hot = self.to_one_hot_vector(label, num)
		if is_root_proc():
			state_compressed = torch.cat((state[:,:-dim_label],self.policy_loc.embedding(label)),1)
		if is_root2_proc():
			state_compressed = state

		return state_compressed


	def postprocess_episodes(self):
		self.policy_episodes = []
		self.disc_episodes = []
		for epi in self.episode_buffers[:-1]:
			s, a, rg, ss1, v, l = Sample(*zip(*epi))
			ss1 = np.vstack(ss1)
			r = self.disc(ss1)

			rg = np.array(rg)

			if self.enable_goal:
				r = r*rg

			policy_epi = {}
			policy_epi['STATES'] = np.vstack(s)
			policy_epi['ACTIONS'] = np.vstack(a)
			policy_epi['REWARD_GOALS'] = rg.reshape(-1)
			policy_epi['REWARDS'] = r.reshape(-1)
			policy_epi['VF_PREDS'] = np.vstack(v).reshape(-1)
			policy_epi['LOG_PROBS'] = np.vstack(l).reshape(-1)

			td_gae = self.policy.compute_ppo_td_gae(policy_epi)

			for key, item in td_gae.items():
				policy_epi[key] = item
			self.policy_episodes.append(policy_epi)

			disc_epi = {}
			disc_epi['STATES_AGENT'] = ss1
			disc_epi['STATES_EXPERT'] = self.sample_states_expert(len(ss1))

			self.disc_episodes.append(disc_epi)

		self.episode_buffers = self.episode_buffers[-1:]

	def gather_episodes(self):
		self.policy_episode_list = gather(self.policy_episodes, root=get_root_proc())
		self.disc_episode_list = gather(self.disc_episodes, root=get_root2_proc())

	def concat_samples(self, episode_list):
		samples = []
		for epis in episode_list:
			for epi in epis:
				samples.append(epi)
		if len(samples) == 0:
			return False

		self.samples = {}
		for key, item in samples[0].items():
			self.samples[key] = []

		for sample in samples:
			for key, item in sample.items():
				self.samples[key].append(item)

		for key in self.samples.keys():
			self.samples[key] = np.concatenate(self.samples[key])

		self.samples['NUM_EPISODES'] = len(samples)
		return True
	
	def standarize_samples(self):
		self.samples['ADVANTAGES'] = (self.samples['ADVANTAGES'] - self.samples['ADVANTAGES'].mean())/(1e-4 + self.samples['ADVANTAGES'].std())

	def update_filter(self):
		if is_root_proc():
			dim_label = self.env.get_dim_state_label()
			self.samples['STATES'][:,:-dim_label] = self.policy_loc.state_filter(self.samples['STATES'][:,:-dim_label])

		if is_root2_proc():
			n = len(self.samples['STATES_EXPERT'])
			dim_label = self.env.get_dim_state_label()
			state = self.disc_loc.state_filter(np.vstack([self.samples['STATES_EXPERT'][:,:-dim_label], self.samples['STATES_AGENT'][:,:-dim_label]]))
			
			label = np.vstack([self.samples['STATES_EXPERT'][:,-dim_label:], self.samples['STATES_AGENT'][:,-dim_label:]])
			
			self.samples['STATES_EXPERT'] = state[:n]
			self.samples['STATES_AGENT'] = state[n:]
			
			self.samples['LABELS_EXPERT'] = label[:n]
			self.samples['LABELS_AGENT'] = label[n:]


	def generate_shuffle_indices(self, batch_size, minibatch_size):
		n = batch_size
		m = minibatch_size
		p = np.random.permutation(n)

		r = m - n%m
		if r>0:
			p = np.hstack([p,np.random.randint(0,n,r)])

		p = p.reshape(-1,m)
		return p

	def optimize(self):
		if is_root_proc():
			n = len(self.samples['STATES'])
			''' Policy '''
			log = {}

			log['mean_episode_len'] = len(self.samples['REWARDS'])/self.samples['NUM_EPISODES']
			log['mean_episode_reward'] = np.mean(self.samples['REWARDS'])
			log['mean_episode_reward_goal'] = np.mean(self.samples['REWARD_GOALS'])

			self.samples['STATES'] = self.policy_loc.convert_to_tensor(self.samples['STATES'])
			self.samples['ACTIONS'] = self.policy_loc.convert_to_tensor(self.samples['ACTIONS'])
			self.samples['VF_PREDS'] = self.policy_loc.convert_to_tensor(self.samples['VF_PREDS'])
			self.samples['LOG_PROBS'] = self.policy_loc.convert_to_tensor(self.samples['LOG_PROBS'])
			self.samples['ADVANTAGES'] = self.policy_loc.convert_to_tensor(self.samples['ADVANTAGES'])
			self.samples['VALUE_TARGETS'] = self.policy_loc.convert_to_tensor(self.samples['VALUE_TARGETS'])

			self.samples['STATES'] = self.compress_label(self.samples['STATES'])

			for _ in range(self.num_sgd_iter):
				minibatches = self.generate_shuffle_indices(n, self.sgd_minibatch_size)
				for minibatch in minibatches:
					states = self.samples['STATES'][minibatch]
					actions = self.samples['ACTIONS'][minibatch]
					vf_preds = self.samples['VF_PREDS'][minibatch]
					log_probs = self.samples['LOG_PROBS'][minibatch]
					advantages = self.samples['ADVANTAGES'][minibatch]
					value_targets = self.samples['VALUE_TARGETS'][minibatch]

					self.policy_loc.compute_loss(states, actions, vf_preds, log_probs, advantages, value_targets)
					self.policy_loc.backward_and_apply_gradients()

			log['std'] = np.mean(np.exp(self.policy_loc.model.policy_fn[-1].log_std.cpu().detach().numpy()))
			self.state_dict['num_iterations_so_far'] += 1
			self.state_dict['num_samples_so_far'] += len(self.samples['REWARDS'])
			
			return log
		if is_root2_proc():
			n = len(self.samples['STATES_EXPERT'])
			''' Discriminator '''
			self.samples['STATES_EXPERT'] = self.disc_loc.convert_to_tensor(self.samples['STATES_EXPERT'])
			self.samples['STATES_EXPERT2'] = self.disc_loc.convert_to_tensor(self.samples['STATES_EXPERT'])
			self.samples['STATES_AGENT'] = self.disc_loc.convert_to_tensor(self.samples['STATES_AGENT'])
			
			disc_loss = 0.0
			disc_grad_loss = 0.0
			disc_classify_loss = 0.0
			expert_accuracy = 0.0
			agent_accuracy = 0.0

			for _ in range(self.num_disc_sgd_iter):
				minibatches = self.generate_shuffle_indices(n, self.sgd_minibatch_size)
				for minibatch in minibatches:
					states_expert = self.samples['STATES_EXPERT'][minibatch]
					states_expert2 = self.samples['STATES_EXPERT2'][minibatch]
					states_agent = self.samples['STATES_AGENT'][minibatch]

					labels_expert = self.samples['LABELS_EXPERT'][minibatch]
					labels_agent = self.samples['LABELS_AGENT'][minibatch]

					self.disc_loc.compute_loss(states_expert, states_expert2, states_agent, labels_expert, labels_agent)
					disc_loss += self.disc_loc.loss.detach()
					disc_grad_loss += self.disc_loc.grad_loss.detach()
					disc_classify_loss += self.disc_loc.classify_loss.detach()
					expert_accuracy += self.disc_loc.expert_accuracy.detach()
					agent_accuracy += self.disc_loc.agent_accuracy.detach()
					self.disc_loc.backward_and_apply_gradients()
			
			log = {}
			log['disc_loss'] = disc_loss.cpu().numpy()
			log['disc_grad_loss'] = disc_grad_loss.cpu().numpy()
			log['disc_classify_loss'] = disc_classify_loss.cpu().numpy()
			log['expert_accuracy'] = expert_accuracy.cpu().numpy()/n/self.num_disc_sgd_iter
			log['agent_accuracy'] = agent_accuracy.cpu().numpy()/n/self.num_disc_sgd_iter
			return log

	def print_log(self, t_sample, log_policy, log_disc, writer = None):
		def time_to_hms(t):
			h = int((t)//3600.0)
			m = int((t)//60.0)
			s = int((t))
			m = m - h*60
			s = t
			s = s - h*3600 - m*60
			return h,m,s
		
		h,m,s=time_to_hms(self.state_dict['elapsed_time'])
		end = '\n'
		print('# {}, {}h:{}m:{:.1f}s ({:.1f}s [s:{:.2f} p:{:.2f}, d:{:.2f}]s) '.format(self.state_dict['num_iterations_so_far'],h,m,s, t_sample + np.max([log_policy['t'], log_disc['t']]), t_sample, log_policy['t'], log_disc['t']),end=end)
		print('policy   len : {:.1f}, rew : {:.3f}, rew_goal : {:.3f}, std : {:.3f} samples : {:,}'.format(log_policy['mean_episode_len'],
																						log_policy['mean_episode_reward'],
																						log_policy['mean_episode_reward_goal'],
																						log_policy['std'],
																						self.state_dict['num_samples_so_far']))
		print('discriminator loss : {:.3f} grad_loss : {:.3f} classify_loss : {:.3f} acc_expert : {:.3f}, acc_agent : {:.3f}'.format(log_disc['disc_loss'],
																						log_disc['disc_grad_loss'],
																						log_disc['disc_classify_loss'],
																						log_disc['expert_accuracy'],
																						log_disc['agent_accuracy']))
		if writer is not None:
			writer.add_scalar('policy/episode_len',log_policy['mean_episode_len'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('policy/reward_mean',log_policy['mean_episode_reward'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('policy/reward_mean_goal',log_policy['mean_episode_reward_goal'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/loss',log_disc['disc_loss'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/grad_loss',log_disc['disc_grad_loss'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/classify_loss',log_disc['disc_classify_loss'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/expert_accuracy',log_disc['expert_accuracy'],
				self.state_dict['num_samples_so_far'])

			writer.add_scalar('discriminator/agent_accuracy',log_disc['agent_accuracy'],
				self.state_dict['num_samples_so_far'])

	def save(self, path):
		cond0 = self.state_dict['num_iterations_so_far'] % self.save_iteration[0] == 0
		cond1 = self.state_dict['num_iterations_so_far'] % self.save_iteration[1] == 0

		if cond0 or cond1:
			state = {}
			
			state['policy_state_dict'] = self.policy.state_dict()
			state['discriminator_state_dict'] = self.disc.state_dict()

			for key, item in self.state_dict.items():
				state[key] = item

			if cond0:
				torch.save(state, os.path.join(path,'current.pt'))
				print('save at {}'.format(os.path.join(path,'current.pt')))
			if cond1:
				torch.save(state, os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt'))
				print('save at {}'.format(os.path.join(path,str(math.floor(self.state_dict['num_samples_so_far']/1e6))+'.pt')))

	def load(self, path):
		if is_root_proc():
			print(f'load {path}')
			state = torch.load(path)
			self.policy_loc.load_state_dict(state['policy_state_dict'])
			for key in self.state_dict.keys():
				self.state_dict[key] = state[key]
		elif is_root2_proc():
			state = torch.load(path)
			self.disc_loc.load_state_dict(state['discriminator_state_dict'])

			