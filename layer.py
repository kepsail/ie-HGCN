import torch
import torch.nn as nn
import torch.nn.functional as F



class HeteGCNLayer(nn.Module):
	def __init__(self, net_schema, in_layer_shape, out_layer_shape):
		super(HeteGCNLayer, self).__init__()
		
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schema:
			self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k])


	def forward(self, x_dict, adj_dict):
		
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k] = self.hete_agg[k](k, x_dict, adj_dict[k])

		return ret_x_dict



class HeteAggregateLayer(nn.Module):
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape):
		super(HeteAggregateLayer, self).__init__()
		
		self.nb_list = nb_list
		
		self.W_proj = nn.ParameterDict()
		for k in nb_list:
			self.W_proj[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
			nn.init.xavier_uniform_(self.W_proj[k].data, gain=1.414)
		
		self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape))
		nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

		self.w_share = nn.Parameter(torch.FloatTensor(out_shape, out_shape))
		nn.init.xavier_uniform_(self.w_share.data, gain=1.414)
			
		self.w_att = nn.Parameter(torch.FloatTensor(2*out_shape, 1))
		nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

		self.w_cat = nn.Parameter(torch.FloatTensor(2*out_shape, out_shape))
		nn.init.xavier_uniform_(self.w_cat.data, gain=1.414)

		self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
		nn.init.xavier_uniform_(self.bias.data, gain=1.414)


	def forward(self, curr_k, x_dict, adj_dict):
		
		self_ft = torch.mm(x_dict[curr_k], self.w_self)
		self_ft = torch.mm(self_ft, self.w_share)

		nb_ft = {}
		for k in self.nb_list:
			nb_ft[k] = torch.mm(x_dict[k], self.W_proj[k])
			nb_ft[k] = torch.spmm(adj_dict[k], nb_ft[k])
			nb_ft[k] = torch.mm(nb_ft[k], self.w_share)

		agg_nb_ft = torch.zeros(self_ft.shape, device=self.bias.device)

		if len(self.nb_list) < 2:
			agg_nb_ft = list(nb_ft.values())[0]
		else:
			nb_ft_list = list(zip(*nb_ft.items()))			
			a_input = torch.cat([torch.cat(nb_ft_list[1], 0), self_ft.repeat(len(nb_ft_list[1]), 1)], 1)
			e = F.leaky_relu(torch.matmul(a_input, self.w_att))
			attention = F.softmax(e.view(-1, len(nb_ft_list[1])), dim=1)

			for i in range(len(nb_ft_list[1])):
				agg_nb_ft = torch.add(agg_nb_ft, nb_ft_list[1][i].mul(attention[:,i].view(-1,1)))

		output = torch.mm(torch.cat([agg_nb_ft, self_ft], 1), self.w_cat) + self.bias
		
		return output