import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import HeteGCNLayer



class HGCN(nn.Module):
	
	def __init__(self, net_schema, layer_shape, label_keys, type_fusion='att', type_att_size=64):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)
		self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], type_fusion, type_att_size)
		self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], type_fusion, type_att_size)
		

		# self.hgc5 = HeteGCNLayer(net_schema, layer_shape[4], layer_shape[5], type_fusion, type_att_size)
		# self.hgc6 = HeteGCNLayer(net_schema, layer_shape[5], layer_shape[6], type_fusion, type_att_size)
		# self.hgc7 = HeteGCNLayer(net_schema, layer_shape[6], layer_shape[7], type_fusion, type_att_size)
		# self.hgc8 = HeteGCNLayer(net_schema, layer_shape[7], layer_shape[8], type_fusion, type_att_size)
		# self.hgc9 = HeteGCNLayer(net_schema, layer_shape[8], layer_shape[9], type_fusion, type_att_size)


		self.embd2class = nn.ParameterDict()
		self.bias = nn.ParameterDict()
		self.label_keys = label_keys
		for k in label_keys:
			self.embd2class[k] = nn.Parameter(torch.FloatTensor(layer_shape[-2][k], layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)


	def forward(self, ft_dict, adj_dict):

		x_dict = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc2(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc3(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc4(x_dict, adj_dict)


		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)

		# x_dict = self.hgc5(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)

		# x_dict = self.hgc6(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)

		# x_dict = self.hgc7(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)

		# x_dict = self.hgc8(x_dict, adj_dict)
		# x_dict = self.non_linear(x_dict)
		# x_dict = self.dropout_ft(x_dict, 0.5)

		# x_dict = self.hgc9(x_dict, adj_dict)




		logits = {}
		embd = {}
		for k in self.label_keys:
			embd[k] = x_dict[k]
			logits[k] = torch.mm(x_dict[k], self.embd2class[k]) + self.bias[k]
		return logits, embd


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.elu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict
