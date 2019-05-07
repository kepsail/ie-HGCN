import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_2 import HeteGCNLayer



class HGCN(nn.Module):
	
	def __init__(self, net_schema, layer_shape, label_keys, layer_fusion='none', type_fusion='att', type_att_size=64, gcn=True):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size, gcn)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size, gcn)
		self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], type_fusion, type_att_size, gcn)
		self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], type_fusion, type_att_size, gcn)
		
		self.label_keys = label_keys
		self.layer_fusion = layer_fusion


		if layer_fusion == 'cat':
		
			self.pred_x3 = nn.ParameterDict()
			self.pred_x4 = nn.ParameterDict()
			self.cat_x345 = nn.ParameterDict()
			self.bias= nn.ParameterDict()

			for k in label_keys:
				self.pred_x3[k] = nn.Parameter(torch.FloatTensor(layer_shape[2][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.pred_x3[k].data, gain=1.414)
				self.pred_x4[k] = nn.Parameter(torch.FloatTensor(layer_shape[3][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.pred_x4[k].data, gain=1.414)
				self.cat_x345[k] = nn.Parameter(torch.FloatTensor(layer_shape[4][k]*3, layer_shape[4][k]))
				nn.init.xavier_uniform_(self.cat_x345[k].data, gain=1.414)
				self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[4][k]))
				nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)


		elif layer_fusion == 'att':

			self.pred_x3 = nn.ParameterDict()
			self.pred_x4 = nn.ParameterDict()
			self.pred_x5 = nn.ParameterDict()
			self.w_query = nn.ParameterDict()
			self.w_keys = nn.ParameterDict()
			self.bias = nn.ParameterDict()

			for k in label_keys:

				self.pred_x3[k] = nn.Parameter(torch.FloatTensor(layer_shape[2][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.pred_x3[k].data, gain=1.414)
				self.pred_x4[k] = nn.Parameter(torch.FloatTensor(layer_shape[3][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.pred_x4[k].data, gain=1.414)
				self.pred_x5[k] = nn.Parameter(torch.FloatTensor(layer_shape[4][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.pred_x5[k].data, gain=1.414)

				self.w_query[k] = nn.Parameter(torch.FloatTensor(layer_shape[4][k], 1))
				nn.init.xavier_uniform_(self.w_query[k].data, gain=1.414)
				self.w_keys[k] = nn.Parameter(torch.FloatTensor(layer_shape[4][k], layer_shape[4][k]))
				nn.init.xavier_uniform_(self.w_keys[k].data, gain=1.414)

				self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[4][k]))
				nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)



	def forward(self, ft_dict, adj_dict):

		x_dict = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x3_dict = self.hgc2(x_dict, adj_dict)
		x_dict = self.non_linear(x3_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x4_dict = self.hgc3(x_dict, adj_dict)
		x_dict = self.non_linear(x4_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x5_dict = self.hgc4(x_dict, adj_dict)


		if self.layer_fusion == 'none':
			return x5_dict
	

		elif self.layer_fusion == 'cat':
			output = {}
			for k in self.label_keys:
				x3_dict[k] = F.dropout(x3_dict[k], 0.5, training=self.training)
				x4_dict[k] = F.dropout(x4_dict[k], 0.5, training=self.training)
				x3_dict[k] = torch.mm(x3_dict[k], self.pred_x3[k])
				x4_dict[k] = torch.mm(x4_dict[k], self.pred_x4[k])
				output[k] = torch.mm(torch.cat([x3_dict[k], x4_dict[k], x5_dict[k]], 1), self.cat_x345[k]) + self.bias[k]
			return output


		elif self.layer_fusion == 'att':

			output = {}
			for k in self.label_keys:

				x3_dict[k] = F.dropout(x3_dict[k], 0.5, training=self.training)
				x4_dict[k] = F.dropout(x4_dict[k], 0.5, training=self.training)
				
				x3_dict[k] = torch.mm(x3_dict[k], self.pred_x3[k])
				x4_dict[k] = torch.mm(x4_dict[k], self.pred_x4[k])
				x5_dict[k] = torch.mm(x5_dict[k], self.pred_x5[k])
				layer_cat =torch.cat([x3_dict[k].unsqueeze(1), x4_dict[k].unsqueeze(1), x5_dict[k].unsqueeze(1)], 1)
		
				att_keys = torch.matmul(layer_cat, self.w_keys[k])
				att_keys = F.dropout(att_keys, 0.3, training=self.training)
				attention = F.softmax(torch.matmul(att_keys, self.w_query[k]), dim=1)
				output[k] = layer_cat.mul(attention).sum(1) + self.bias[k]
			
				print('label_keys: ', k, '\t3,4,5: ', attention.mean(0).tolist())
			return output


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
