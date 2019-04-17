import torch
import torch.nn as nn
import torch.nn.functional as F

from v2_layer import HeteGCNLayer, ProjLayer


class HGCN(nn.Module):
	
	def __init__(self, net_schema, layer_shape, label_keys, layer_att_size=128, proj_ft=False, layer_fusion='none'):
		super(HGCN, self).__init__()
		
		self.label_keys = label_keys
		self.proj_ft = proj_ft
		self.layer_fusion = layer_fusion

		if proj_ft:
			self.proj = ProjLayer(layer_shape[0], max(layer_shape[1].values()), non_linear=False, dropout=0.5)
			layer_shape_0 = layer_shape[0]
			layer_shape_0.update(dict.fromkeys(layer_shape[0].keys(), max(layer_shape[1].values())))

		if layer_fusion == 'none':
			self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], non_linear=True, dropout=0.5)
			self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], non_linear=True, dropout=0.5)
			self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], non_linear=True, dropout=0.5)
			self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], non_linear=False, dropout=0.5)
		
		else:
			self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], non_linear=True, dropout=0.5)
			self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], non_linear=True, dropout=0.5)
			self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], non_linear=True, dropout=0.5)
			self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], non_linear=True, dropout=0.5)

			if layer_fusion == 'cat':

				self.pred_linear_1_dict = nn.ModuleDict()
				self.pred_linear_2_dict = nn.ModuleDict()
				for k in label_keys:
					layer_cat_dim = layer_shape[2][k]+layer_shape[3][k]+layer_shape[4][k]
					self.pred_linear_1_dict[k] = nn.Linear(layer_cat_dim, int(layer_cat_dim/6))
					self.pred_linear_2_dict[k] = nn.Linear(int(layer_cat_dim/6), layer_shape[5][k])

			elif layer_fusion == 'att':
				self.layer_att_linear_dict = nn.ModuleDict()
				self.w_layer_att_dict = nn.ParameterDict()
				self.pred_att_dict = nn.ModuleDict()
				for k in label_keys:
					self.layer_att_linear_dict[k] = nn.Linear(layer_shape[4][k], layer_att_size)
					self.w_layer_att_dict[k]= nn.Parameter(torch.FloatTensor(layer_att_size, 1))
					nn.init.xavier_uniform_(self.w_layer_att_dict[k].data, gain=1.414)
					self.pred_att_dict[k] = nn.Linear(layer_shape[4][k], layer_shape[5][k])


	def forward(self, ft_dict, adj_dict):

		if self.proj_ft:
			x1_dict = self.proj(ft_dict)
		else:
			x1_dict = ft_dict

		x2_dict = self.hgc1(x1_dict, adj_dict)
		x3_dict = self.hgc2(x2_dict, adj_dict)
		x4_dict = self.hgc3(x3_dict, adj_dict)
		x5_dict = self.hgc4(x4_dict, adj_dict)

		if self.layer_fusion == 'none':
			return x5_dict
		
		elif self.layer_fusion == 'cat':
			output = {}
			for k in self.label_keys:
				output[k] = self.pred_linear_1_dict[k](torch.cat([x3_dict[k], x4_dict[k], x5_dict[k]], 1))
				output[k] = self.pred_linear_2_dict[k](output[k])
			return output
	
		elif self.layer_fusion == 'att':
			output = {}
			for k in self.label_keys:
				layer_cat = torch.cat([x3_dict[k].unsqueeze(1), x4_dict[k].unsqueeze(1), x5_dict[k].unsqueeze(1)] ,1)
				layer_cat_att = self.layer_att_linear_dict[k](layer_cat)
				layer_attention = F.softmax(layer_cat_att.matmul(self.w_layer_att_dict[k]), dim=1)
				output[k] = self.pred_att_dict[k](layer_cat.mul(layer_attention).sum(1))
			return output