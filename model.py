import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import HeteGCNLayer



class HGCN(nn.Module):
	def __init__(self, net_schema, layer_shape, dropout):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], dropout)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], dropout)
		# self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], dropout)
		# self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], dropout)
		
		self.net_schema = net_schema


	def forward(self, ft_dict, adj_dict):

		x_dict = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc2(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)
		
		# x_dict = self.hgc3(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.2)
		
		# x_dict = self.hgc4(x_dict, adj_dict)
		return x_dict

	def non_linear(self, x_dict):
		for k in x_dict:
			x_dict[k] = F.relu(x_dict[k])
		return x_dict

	def dropout_ft(self, x_dict, dropout):
		for k in x_dict:
			x_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return x_dict
