import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import HeteGCNLayer



class HGCN(nn.Module):
	
	def __init__(self, net_schema, layer_shape, label_keys, layer_fusion='none'):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1])
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2])
		self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3])
		self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4])
		
		self.net_schema = net_schema
		self.label_keys = label_keys
		self.layer_fusion = layer_fusion

		if layer_fusion == 'cat':
			self.pred_linear_1_dict = nn.ModuleDict()
			self.pred_linear_2_dict = nn.ModuleDict()
			for k in label_keys:
				layer_cat_dim = layer_shape[2][k]+layer_shape[4][k]
				self.pred_linear_1_dict[k] = nn.Linear(layer_cat_dim, int(layer_cat_dim/4))
				self.pred_linear_2_dict[k] = nn.Linear(int(layer_cat_dim/4), layer_shape[5][k])


	def forward(self, ft_dict, adj_dict):

		if self.layer_fusion == 'none':

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

			return x_dict


		elif self.layer_fusion == 'cat':

			x_dict = self.layer_norm(ft_dict)
			x_dict = self.hgc1(x_dict, adj_dict)
			x_dict = self.non_linear(x_dict)
			y_dict = self.dropout_ft(x_dict, 0.5)

			x_dict = self.layer_norm(y_dict)
			x3_dict = self.hgc2(x_dict, adj_dict)
			x_dict = self.non_linear(x3_dict)
			x_dict = self.dropout_ft(x_dict, 0.5)
			y_dict = self.res_conn(y_dict, x_dict)
			
			x_dict = self.layer_norm(y_dict)
			x_dict = self.hgc3(x_dict, adj_dict)
			x_dict = self.non_linear(x_dict)
			x_dict = self.dropout_ft(x_dict, 0.5)
			y_dict = self.res_conn(y_dict, x_dict)
			
			x_dict = self.layer_norm(y_dict)
			x5_dict = self.hgc4(x_dict, adj_dict)

			output = {}
			for k in self.label_keys:
				layer_cat = torch.cat([x3_dict[k], x5_dict[k]], 1)
				layer_cat = F.dropout(layer_cat, 0.5, training=self.training)
				output[k] = self.pred_linear_1_dict[k](layer_cat)
				output[k] = F.dropout(output[k], 0.5, training=self.training)
				output[k] = self.pred_linear_2_dict[k](output[k])
			return output


	def res_conn(self, x_dict, y_dict):
		z_dict = {}
		for k in x_dict:
			z_dict[k] = x_dict[k] + y_dict[k]
		return z_dict


	def layer_norm(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.layer_norm(x_dict[k], x_dict[k].size())
		return y_dict


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.relu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict
