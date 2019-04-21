import torch
import torch.nn as nn
import torch.nn.functional as F



class ProjLayer(nn.Module):

	def __init__(self, in_layer_shape, out_shape, non_linear=False, dropout=0.5):
		super(ProjLayer, self).__init__()
		
		self.proj_linear_dict = nn.ModuleDict()
		for k in in_layer_shape.keys():
			self.proj_linear_dict[k] = nn.Linear(in_layer_shape[k], out_shape)

		self.non_linear = non_linear	
		if non_linear:
			self.non_linear_act = nn.ReLU()

		self.dropout_layer = nn.Dropout(p=dropout)


	def forward(self, ft_dict):
		x_dict = {}
		for k in ft_dict.keys():
			x_dict[k] = self.proj_linear_dict[k](ft_dict[k])
			if self.non_linear:
				x_dict[k] = self.non_linear_act(x_dict[k])
			x_dict[k] = self.dropout_layer(x_dict[k])

		return x_dict



class HeteGCNLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, out_layer_shape, non_linear=True, dropout=0.5):
		super(HeteGCNLayer, self).__init__()
		
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_block = HeteLayer(net_schema, in_layer_shape)
		self.hete_norm = nn.ModuleDict()
		for k in net_schema.keys():
			self.hete_norm[k] = nn.LayerNorm([in_layer_shape[k]])
		
		self.ff_block = FeedForwardLayer(in_layer_shape, out_layer_shape, non_linear=non_linear)
		self.ff_norm = nn.ModuleDict()
		self.ff_res_linear = nn.ModuleDict()
		for k in net_schema.keys():
			self.ff_norm[k] = nn.LayerNorm([out_layer_shape[k]])
			if in_layer_shape[k] != out_layer_shape[k]:
				self.ff_res_linear[k] = nn.Linear(in_layer_shape[k], out_layer_shape[k])

		self.dropout_layer = nn.Dropout(p=dropout)


	def forward(self, x_dict, adj_dict):

		y_dict = self.hete_block(x_dict, adj_dict)
		for k in y_dict.keys():
			y_dict[k] = y_dict[k] + x_dict[k]
			y_dict[k] = self.hete_norm[k](y_dict[k])

		z_dict = self.ff_block(y_dict)
		for k in z_dict.keys():
			if z_dict[k].shape[-1] == y_dict[k].shape[-1]:
				z_dict[k] = z_dict[k] + y_dict[k]
			else:
				z_dict[k] = z_dict[k] + self.ff_res_linear[k](y_dict[k])
			z_dict[k] = self.ff_norm[k](z_dict[k])

		for z in z_dict.keys():
			z_dict[k] = self.dropout_layer(z_dict[k])
		return z_dict



class HeteLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, dropout=0.1):
		super(HeteLayer, self).__init__()

		self.hete_block = nn.ModuleDict()
		for k in net_schema.keys():
			self.hete_block[k] = HeteBlock(k, net_schema[k], in_layer_shape, in_layer_shape[k])

		self.dropout_layer = nn.Dropout(p=dropout)


	def forward(self, x_dict, adj_dict):
		
		y_dict = {}
		for k in self.hete_block.keys():
			y_dict[k] = self.hete_block[k](x_dict, adj_dict[k])
			y_dict[k] = self.dropout_layer(y_dict[k])
		
		return y_dict



class HeteBlock(nn.Module):
	
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, nb_att_size=64, rela_att_size=64, nb_att='none', rela_att='dot'):
		super(HeteBlock, self).__init__()
		
		self.curr_k = curr_k
		self.nb_list = nb_list
		self.nb_att = nb_att
		self.rela_att = rela_att
		
		self.W_rel = nn.ParameterDict()
		for k in nb_list:
			self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
			nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
		

		if nb_att == 'cat':
			self.W_nb_att = nn.ParameterDict()
			for k in nb_list:
				self.W_nb_att = nn.Parameter(torch.FloatTensor(2*out_shape, 1))
				nn.init.xavier_uniform_(self.W_nb_att.data, gain=1.414)
		
		elif nb_att == 'dot':
			self.nb_query_linear = nn.Linear(out_shape, nb_att_size)
			self.nb_keys_linear_dict = nn.ModuleDict()
			for k in nb_list:
				self.nb_keys_linear_dict[k] = nn.Linear(out_shape, nb_att_size)


		if rela_att == 'cat':
			self.w_rela_att = nn.Parameter(torch.FloatTensor(2*out_shape, 1))
			nn.init.xavier_uniform_(self.w_rela_att.data, gain=1.414)
	
		elif rela_att == 'dot':
			self.rela_query_linear = nn.Linear(out_shape, rela_att_size)
			self.rela_keys_linear = nn.Linear(out_shape, rela_att_size)


		self.nb_cat_self = nn.Linear(2*out_shape, out_shape)



	def forward(self, x_dict, adj_dict):
		
		nb_ft = {}
		if self.nb_att == 'none':
			for k in self.nb_list:
				nb_ft[k] = torch.mm(x_dict[k], self.W_rel[k])
				nb_ft[k] = torch.spmm(adj_dict[k], nb_ft[k])

		elif self.nb_att == 'dot':
			att_query = self.nb_query_linear(x_dict[self.curr_k])
			for k in self.nb_list:
				nb_ft[k] = torch.mm(x_dict[k], self.W_rel[k])
				att_keys = self.nb_keys_linear_dict[k](nb_ft[k])
				nb_att = torch.mm(att_query, att_keys.transpose(0,1))
				zero_vec = -9e15*torch.ones_like(nb_att)
				nb_attention = torch.where(adj_dict[k] > 0, nb_att, zero_vec)
				nb_attention = F.softmax(nb_attention, dim=1)
				nb_ft[k] = torch.mm(nb_attention, nb_ft[k])

		elif self.nb_att == 'cat':
			pass


		if len(self.nb_list) < 2:
			agg_nb_ft = list(nb_ft.values())[0]
		else:
			nb_ft_list = list(nb_ft.values())			
			
			if self.rela_att == 'dot':
				att_query = self.rela_query_linear(x_dict[self.curr_k]).unsqueeze(-1)
				att_keys = self.rela_keys_linear(torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1))
				rela_attention = F.softmax(att_keys.matmul(att_query).squeeze(), dim=1)
				agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(rela_attention.unsqueeze(-1)).sum(1)
			
			elif self.rela_att == 'cat':
				a_input = torch.cat([torch.cat(nb_ft_list, 0), x_dict[self.curr_k].repeat(len(nb_ft_list), 1)], 1)
				e = F.leaky_relu(torch.matmul(a_input, self.w_rela_att))
				rela_attention = F.softmax(e.view(-1, len(nb_ft_list)), dim=1)
				agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(rela_attention.unsqueeze(-1)).sum(1)


		output = self.nb_cat_self(torch.cat([agg_nb_ft, x_dict[self.curr_k]], 1))
		return output



class FeedForwardLayer(nn.Module):

	def __init__(self, in_layer_shape, out_layer_shape, non_linear=True, dropout=0.1):
		super(FeedForwardLayer, self).__init__()
		
		self.ff_linear_dict = nn.ModuleDict()
		for k in in_layer_shape.keys():
			self.ff_linear_dict[k] = nn.Linear(in_layer_shape[k], out_layer_shape[k])

		self.non_linear = non_linear	
		if non_linear:
			self.non_linear_act = nn.ReLU()

		self.dropout_layer = nn.Dropout(p=dropout)


	def forward(self, x_dict):
		
		for k in x_dict.keys():
			x_dict[k] = self.ff_linear_dict[k](x_dict[k])
			if self.non_linear:
				x_dict[k] = self.non_linear_act(x_dict[k])
			x_dict[k] = self.dropout_layer(x_dict[k])

		return x_dict
