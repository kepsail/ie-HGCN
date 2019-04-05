import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import HeteGCNLayer



class HGCN(nn.Module):
	def __init__(self, net_schema, layer_shape, label_keys):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1])
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2])
		self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3])
		self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4])
		
		self.net_schema = net_schema
		self.label_keys = label_keys

		# self.W_out = nn.ParameterDict()
		# for k in label_keys:
		# 	self.W_out[k] = nn.Parameter(torch.FloatTensor(layer_shape[2][k]+layer_shape[3][k]+layer_shape[4][k], layer_shape[4][k]))
		# 	nn.init.xavier_uniform_(self.W_out[k].data, gain=1.414)

		# self.bias_out = nn.Parameter(torch.FloatTensor(1, layer_shape[4][k]))
		# nn.init.xavier_uniform_(self.bias_out.data, gain=1.414)


	def forward(self, ft_dict, adj_dict):

		x_dict = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc2(x_dict, adj_dict)
		x2_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x2_dict, 0.3)
		
		x_dict = self.hgc3(x_dict, adj_dict)
		x3_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x3_dict, 0.2)
		
		x_dict = self.hgc4(x_dict, adj_dict)

		return x_dict
	

		# x4_dict = self.non_linear(x_dict)
		# output = {}
		# for k in self.label_keys:
		# 	output[k] = torch.mm(torch.cat([x2_dict[k], x3_dict[k], x4_dict[k]], 1), self.W_out[k]) + self.bias_out
		# return output


	def non_linear(self, x_dict):
		for k in x_dict:
			x_dict[k] = F.relu(x_dict[k])
		return x_dict


	def dropout_ft(self, x_dict, dropout):
		for k in x_dict:
			x_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return x_dict
