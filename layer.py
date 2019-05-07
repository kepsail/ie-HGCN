import torch
import torch.nn as nn
import torch.nn.functional as F



class HeteGCNLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size, gcn):
		super(HeteGCNLayer, self).__init__()
		
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schema:
			self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k], type_fusion, type_att_size, gcn)


	def forward(self, x_dict, adj_dict):
		
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k] = self.hete_agg[k](x_dict, adj_dict[k])

		return ret_x_dict



class HeteAggregateLayer(nn.Module):
	
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size, gcn):
		super(HeteAggregateLayer, self).__init__()
		
		self.nb_list = nb_list
		self.curr_k = curr_k
		self.gcn = gcn
		self.type_fusion = type_fusion
		
		self.W_rel = nn.ParameterDict()
		for k in nb_list:
			self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
			nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
		
		self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape))
		nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

		self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
		nn.init.xavier_uniform_(self.bias.data, gain=1.414)

		if gcn:
			self.w_cat = nn.Parameter(torch.FloatTensor(2*out_shape, out_shape))
			nn.init.xavier_uniform_(self.w_cat.data, gain=1.414)
	
		if type_fusion == 'att':
			self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
			self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
			self.w_att = nn.Parameter(torch.FloatTensor(2*type_att_size, 1))
			nn.init.xavier_uniform_(self.w_att.data, gain=1.414)


	def forward(self, x_dict, adj_dict):
		
		self_ft = torch.mm(x_dict[self.curr_k], self.w_self)

		nb_ft = {}
		for k in self.nb_list:
			nb_ft[k] = torch.mm(x_dict[k], self.W_rel[k])
			nb_ft[k] = torch.spmm(adj_dict[k], nb_ft[k])

		if len(self.nb_list) < 2:
			agg_nb_ft = list(nb_ft.values())[0]
		else:
			nb_ft_list = list(nb_ft.values())

			if self.type_fusion == 'mean':
				agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)

			elif self.type_fusion == 'att':
				att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
				att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
				att_input = torch.cat([att_keys, att_query], 1)
				att_input = F.dropout(att_input, 0.5, training=self.training)
				e = F.elu(torch.matmul(att_input, self.w_att))
				attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0,1), dim=1)
				agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
				
				# print(list(nb_ft.keys()), attention.mean(0).tolist())
			
		if self.gcn:
			output = torch.mm(torch.cat([agg_nb_ft, self_ft], 1), self.w_cat) + self.bias
		else:
			output = agg_nb_ft + self.bias

		return output