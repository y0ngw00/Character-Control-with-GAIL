import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import time
import copy
import numpy as np

import filter

from misc import *

class DiscriminatorNN(nn.Module):
	def __init__(self, dim_in, num_class, dim_class, model_config):
		nn.Module.__init__(self)

		hiddens = model_config['hiddens']
		activations = model_config['activations']
		init_weights = model_config['init_weights']
		embedding_length = model_config['embedding_length']
		layers = []


		self.dim_in = dim_in - dim_class

		# self.label_embedding = nn.Embedding( embedding_length,  dim_label_out)


		prev_layer_size = self.dim_in
		
		for size, activation, init_weight in zip(hiddens, activations, init_weights):
			layers.append(SlimFC(
				prev_layer_size,
				size,
				xavier_initializer(init_weight),
				activation,
				"SpectralNorm"))
			prev_layer_size = size

		# out_class = layers.append(limFC(
		# 		prev_layer_size,
		# 		num_class,
		# 		xavier_initializer(init_weight),
		# 		activation,
		# 		))
		self.fn = nn.Sequential(*layers)
		self.fn_out = SlimFC(
				prev_layer_size,
				1,
				xavier_initializer(init_weights[-1]),
				activations[-1]
				)
		self.fn_class = SlimFC(
				prev_layer_size,
				num_class,
				xavier_initializer(init_weights[-1]),
				"softmax"
				)
		# self.fn = nn.Sequential(*layers)

		
	def forward(self, x):
		output = self.fn(x)
		out = self.fn_out(output)
		out_class = self.fn_class(output)
		return out,out_class


class Discriminator(object):
	def __init__(self, dim_state,num_class, dim_class, device, model_config, disc_config):
		self.model = DiscriminatorNN(dim_state, num_class,dim_class, model_config)

		self.state_filter = filter.MeanStdRuntimeFilter(self.model.dim_in)
		self.w_grad = disc_config['w_grad']
		self.w_reg = disc_config['w_reg']
		self.w_decay = disc_config['w_decay']
		self.r_scale = disc_config['r_scale']
		self.loss_type = disc_config['loss']

		self.grad_clip = disc_config['grad_clip']

		self.optimizer = optim.Adam(self.model.parameters(),lr=disc_config['lr'])
		self.dim_class = dim_class
		self.num_class = num_class

		self.loss = None
		self.device = device
		self.model.to(self.device)

	def __call__(self, ss1):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)

		ss1_filtered = self.state_filter(ss1[:,:-self.dim_class], update=False)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)

		with torch.no_grad():
			d,l = self.model(ss1_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))

		return d

	def embedding(self, tensor):
		if tensor.dtype == torch.float:
			tensor = tensor.long().to(self.device)
			out = self.model.label_embedding(tensor)
			return out.float().to(self.device)

		else :
			out = self.model.label_embedding(tensor)
			return out.to(self.device)

	def convert_to_tensor(self, arr, type=float):
		if torch.is_tensor(arr):
			return arr.to(self.device)
		tensor = torch.from_numpy(np.asarray(arr))

		if tensor.dtype == torch.double:
			tensor = tensor.float()
		if type=="long":
			tensor = tensor.long()
		return tensor.to(self.device)

	def convert_to_ndarray(self, arr):
		if isinstance(arr, np.ndarray):
			if arr.dtype == np.float64:
				return arr.astype(np.float32)
			return arr
		return arr.cpu().detach().numpy().squeeze()

	
	def compute_loss(self, s_expert, s_expert2, s_agent,l_expert_target, l_agent_target):
		d_expert,l_expert = self.model(s_expert)
		d_agent,l_agent  = self.model(s_agent)
		l_expert_target = torch.squeeze(self.convert_to_tensor(l_expert_target,"long"))
		l_agent_target = torch.squeeze(self.convert_to_tensor(l_agent_target,"long"))

		if self.loss_type == 'hinge loss':
			zero = torch.Tensor([0]).to(self.device)
			loss_pos = 0.5 * torch.mean(torch.max(zero,-d_expert + 1.0))
			loss_neg = 0.5 * torch.mean(torch.max(zero,d_agent  + 1.0))
		else :
			loss_pos = 0.5 * torch.mean(torch.pow(d_expert - 1.0, 2.0))
			loss_neg = 0.5 * torch.mean(torch.pow(d_agent  + 1.0, 2.0))
			''' Compute Accuracy'''
		self.expert_accuracy = torch.sum(d_expert)
		self.agent_accuracy = torch.sum(d_agent)

		CELoss = nn.CrossEntropyLoss()
		loss_expt_class=CELoss(l_expert,l_expert_target)
		loss_agnt_class=CELoss(l_agent,l_agent_target)

		self.loss = 0.5 * (loss_pos + loss_neg)

		self.classify_loss = 0.5* (loss_expt_class+loss_agnt_class)
		self.loss += self.classify_loss

		if self.w_decay>0:
			for i in range(len(self.model.fn)):
				v = self.model.fn[i].model[0].weight
				self.loss += 0.5* self.w_decay * torch.sum(v**2)


		if self.w_reg>0:
			v = self.model.fn_out.model[0].weight
			self.loss += 0.5* self.w_reg * torch.sum(v**2)

		if self.w_grad>0:
			batch_size = s_expert.size()[0]
			s_expert2.requires_grad = True
			d_expert2,l_expert2 = self.model(s_expert2)
			
			grad = torch.autograd.grad(outputs=d_expert2, 
										inputs=s_expert2,
										grad_outputs=torch.ones(d_expert2.size()).to(self.device),
										create_graph=True,
										retain_graph=True)[0]
			
			self.grad_loss = 0.5 * self.w_grad * torch.mean(torch.sum(torch.pow(grad, 2.0), axis=-1))
			self.loss += self.grad_loss
		else:
			self.grad_loss = self.convert_to_tensor(np.array(0.0))

	def backward_and_apply_gradients(self):
		self.optimizer.zero_grad()
		self.loss.backward(retain_graph = True)
		for param in self.model.parameters():
			if param.grad is not None:
				param.grad.data.clamp_(-self.grad_clip,self.grad_clip)
		self.optimizer.step()
		self.loss = None

	def state_dict(self):
		state = {}
		state['model'] = self.model.state_dict()
		state['optimizer'] = self.optimizer.state_dict()
		state['state_filter'] = self.state_filter.state_dict()

		return state

	def load_state_dict(self, state):
		self.model.load_state_dict(state['model'])
		self.optimizer.load_state_dict(state['optimizer'])
		self.state_filter.load_state_dict(state['state_filter'])

	def compute_reward(self, ss1):
		if len(ss1.shape) == 1:
			ss1 = ss1.reshape(1, -1)
		# ss1_filtered = self.state_filter(ss1)
		# ss1 = self.convert_to_tensor(ss1_filtered)

		ss1_filtered = self.state_filter(ss1[:,:-self.dim_class], update=False)
		ss1_tensor = self.convert_to_tensor(ss1_filtered)


		d,l = self.model(ss1_tensor)
		d = self.convert_to_ndarray(d)
		d = np.clip(d, -1.0, 1.0)
		d = self.r_scale*(1.0 - 0.25*(d-1)*(d-1))
		return d

'''Below function do not use when training'''
import importlib.util

def build_discriminator(dim_state, num_class,dim_class,state_experts, config):
	return Discriminator(dim_state, num_class,dim_class, torch.device(0), config['discriminator_model'], config['discriminator'])

def load_discriminator(discriminator, checkpoint):
	state = torch.load(checkpoint)
	state = state['discriminator_state_dict']
	discriminator.load_state_dict(state)