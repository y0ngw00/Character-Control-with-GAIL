import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import filter

import time

def xavier_initializer(gain=1.0):
	def initializer(tensor):
		torch.nn.init.xavier_uniform_(tensor,gain=gain)
	return initializer
def uniform_initializer(lo=-1.0,up=1.0):
	def initializer(tensor):
		torch.nn.init.uniform_(tensor,a=lo,b=up)
	return initializer
	
class SlimFC(nn.Module):
	def __init__(self,
				 in_size,
				 out_size,
				 initializer,
				 activation=None,
				 Norm = "None"):
		super(SlimFC, self).__init__()
		layers = []
		linear = nn.Linear(in_size, out_size)
		if Norm == "SpectralNorm":
			linear = nn.utils.spectral_norm(linear)
		initializer(linear.weight)
		nn.init.constant_(linear.bias, 0.0)
		layers.append(linear)
		if activation == "relu":
			layers.append(nn.ReLU())
		elif activation == "softmax":
			layers.append(nn.Softmax())
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)

class AppendLogStd(nn.Module):
	def __init__(self, init_log_std, dim , fixed_grad = True):
		super().__init__()
		self.log_std = torch.nn.Parameter(
			torch.as_tensor([init_log_std] * dim).type(torch.float32))
		self.register_parameter("log_std", self.log_std)
		self.log_std.requires_grad = not fixed_grad
	def set_value(self, val):

		self.log_std.data = torch.full(self.log_std.shape,np.log(val),device=self.log_std.device)
	def forward(self, x):
		x = torch.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=-1)
		return x



def to_one_hot_vector(array, dim):
	v = (array.squeeze()).astype(int)
	out = np.eye(dim)[v]
	return out
