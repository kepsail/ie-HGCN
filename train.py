import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

from util import *
from model import HGCN



def train(epoch):

	model.train()
	optimizer.zero_grad()
	output = model(ft_dict, adj_dict)



	# m_logits = torch.sigmoid(output['m'])
	# idx_train = label['m'][1]
	# x_train = m_logits[idx_train]
	# y_train = label['m'][0][idx_train].type_as(x_train)
	# loss_train = F.binary_cross_entropy(x_train, y_train, reduction='none').mean(0).sum()
	# roc_train = roc_auc_score(y_train.data.cpu(), x_train.data.cpu())
	# ap_train = average_precision_score(y_train.data.cpu(), x_train.data.cpu(), average='micro')
	# f1_micro_train = f1_score(y_train.data.cpu(), x_train.round().data.cpu(), average='micro')
	# f1_macro_train = f1_score(y_train.data.cpu(), x_train.round().data.cpu(), average='macro')




	# m_logits = F.log_softmax(output['m'], dim=1)
	# idx_train_m = label['m'][1]
	# x_train_m = m_logits[idx_train_m]
	# y_train_m = label['m'][0][idx_train_m]
	# loss_train = F.nll_loss(x_train_m, y_train_m)
	# f1_micro_train_m = f1_score(y_train_m.data.cpu(), x_train_m.data.cpu().argmax(1), average='micro')
	# f1_macro_train_m = f1_score(y_train_m.data.cpu(), x_train_m.data.cpu().argmax(1), average='macro')

	p_logits = F.log_softmax(output['p'], dim=1)
	idx_train_p = label['p'][1]
	x_train_p = p_logits[idx_train_p]
	y_train_p = label['p'][0][idx_train_p]
	# loss_train_p = F.nll_loss(x_train_p, y_train_p)
	loss_train = F.nll_loss(x_train_p, y_train_p)
	f1_micro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='micro')
	f1_macro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='macro')

	# a_logits = F.log_softmax(output['a'], dim=1)
	# idx_train_a = label['a'][1]
	# x_train_a = a_logits[idx_train_a]
	# y_train_a = label['a'][0][idx_train_a]
	# # loss_train_a = F.nll_loss(x_train_a, y_train_a)
	# loss_train = F.nll_loss(x_train_a, y_train_a)
	# f1_micro_train_a = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='micro')
	# f1_macro_train_a = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='macro')

	# c_logits = F.log_softmax(output['c'], dim=1)
	# idx_train_c = label['c'][1]
	# x_train_c = c_logits[idx_train_c]
	# y_train_c = label['c'][0][idx_train_c]
	# loss_train_c = F.nll_loss(x_train_c, y_train_c)
	# f1_micro_train_c = f1_score(y_train_c.data.cpu(), x_train_c.data.cpu().argmax(1), average='micro')
	# f1_macro_train_c = f1_score(y_train_c.data.cpu(), x_train_c.data.cpu().argmax(1), average='macro')
	
	# loss_train = 0.024*loss_train_p + 0.971*loss_train_a + 0.005*loss_train_c


	loss_train.backward()
	optimizer.step()



	'''///////////////// Validatin ///////////////////'''
	# model.eval()
	# output = model(ft_dict, adj_dict)


	# m_logits = torch.sigmoid(output['m'])
	# idx_val = label['m'][2]
	# x_val = m_logits[idx_val]
	# y_val = label['m'][0][idx_val].type_as(x_val)
	# roc_val = roc_auc_score(y_val.data.cpu(), x_val.data.cpu())
	# ap_val = average_precision_score(y_val.data.cpu(), x_val.data.cpu(), average='micro')
	# f1_micro_val = f1_score(y_val.data.cpu(), x_val.round().data.cpu(), average='micro')
	# f1_macro_val = f1_score(y_val.data.cpu(), x_val.round().data.cpu(), average='macro')

	# print(
	# 	  'epoch: {:3d}'.format(epoch),
	# 	  'train loss: {:.4f}'.format(loss_train.item()),
	# 	  'train roc: {:.4f}'.format(roc_train),
	# 	  'train ap: {:.4f}'.format(ap_train),
	# 	  'train micro f1: {:.4f}'.format(f1_micro_train),
	# 	  'train macro f1: {:.4f}'.format(f1_macro_train),
	# 	  'val roc: {:.4f}'.format(roc_val),
	# 	  'val ap: {:.4f}'.format(ap_val),
	# 	  'val micro f1: {:.4f}'.format(f1_micro_val),
	# 	  'val macro f1: {:.4f}'.format(f1_macro_val),
	# 	 )



	# m_logits = F.log_softmax(output['m'], dim=1)
	# idx_val_m = label['m'][2]
	# x_val_m = m_logits[idx_val_m]
	# y_val_m = label['m'][0][idx_val_m]
	# f1_micro_val_m = f1_score(y_val_m.data.cpu(), x_val_m.data.cpu().argmax(1), average='micro')
	# f1_macro_val_m = f1_score(y_val_m.data.cpu(), x_val_m.data.cpu().argmax(1), average='macro')

	p_logits = F.log_softmax(output['p'], dim=1)
	idx_val_p = label['p'][2]
	x_val_p = p_logits[idx_val_p]
	y_val_p = label['p'][0][idx_val_p]
	f1_micro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='micro')
	f1_macro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='macro')

	# a_logits = F.log_softmax(output['a'], dim=1)
	# idx_val_a = label['a'][2]
	# x_val_a = a_logits[idx_val_a]
	# y_val_a = label['a'][0][idx_val_a]
	# f1_micro_val_a = f1_score(y_val_a.data.cpu(), x_val_a.data.cpu().argmax(1), average='micro')
	# f1_macro_val_a = f1_score(y_val_a.data.cpu(), x_val_a.data.cpu().argmax(1), average='macro')

	# c_logits = F.log_softmax(output['c'], dim=1)
	# idx_val_c = label['c'][2]
	# x_val_c = c_logits[idx_val_c]
	# y_val_c = label['c'][0][idx_val_c]
	# f1_micro_val_c = f1_score(y_val_c.data.cpu(), x_val_c.data.cpu().argmax(1), average='micro')
	# f1_macro_val_c = f1_score(y_val_c.data.cpu(), x_val_c.data.cpu().argmax(1), average='macro')
	
	if epoch % 10 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1 p: {:.4f}'.format(f1_micro_train_p.item()),
			  'train macro f1 p: {:.4f}'.format(f1_macro_train_p.item()),
			  # 'train micro f1 m: {:.4f}'.format(f1_micro_train_m.item()),
			  # 'train macro f1 m: {:.4f}'.format(f1_macro_train_m.item()),
			  # 'train micro f1 a: {:.4f}'.format(f1_micro_train_a.item()),
			  # 'train macro f1 a: {:.4f}'.format(f1_macro_train_a.item()),
			  'val micro f1 p: {:.4f}'.format(f1_micro_val_p.item()),
			  'val macro f1 p: {:.4f}'.format(f1_macro_val_p.item()),
			  # 'val micro f1 m: {:.4f}'.format(f1_micro_val_m.item()),
			  # 'val macro f1 m: {:.4f}'.format(f1_macro_val_m.item()),
			  # 'val micro f1 a: {:.4f}'.format(f1_micro_val_a.item()),
			  # 'val macro f1 a: {:.4f}'.format(f1_macro_val_a.item()),
			 )



def test():
	model.eval()
	output = model(ft_dict, adj_dict)



	# m_logits = torch.sigmoid(output['m'])
	# idx_test = label['m'][3]
	# x_test = m_logits[idx_test]
	# y_test = label['m'][0][idx_test].type_as(x_test)
	# roc_test = roc_auc_score(y_test.data.cpu(), x_test.data.cpu())
	# ap_test = average_precision_score(y_test.data.cpu(), x_test.data.cpu(), average='micro')
	# f1_micro_test = f1_score(y_test.data.cpu(), x_test.round().data.cpu(), average='micro')
	# f1_macro_test = f1_score(y_test.data.cpu(), x_test.round().data.cpu(), average='macro')

	# print('\n'
	# 	  'test roc: {:.4f}'.format(roc_test),
	# 	  'test ap: {:.4f}'.format(ap_test),
	# 	  'test micro f1: {:.4f}'.format(f1_micro_test),
	# 	  'test macro f1: {:.4f}'.format(f1_macro_test),
	# 	 )



	# m_logits = F.log_softmax(output['m'], dim=1)
	# idx_test_m = label['m'][3]
	# x_test_m = m_logits[idx_test_m]
	# y_test_m = label['m'][0][idx_test_m]
	# f1_micro_test_m = f1_score(y_test_m.data.cpu(), x_test_m.data.cpu().argmax(1), average='micro')
	# f1_macro_test_m = f1_score(y_test_m.data.cpu(), x_test_m.data.cpu().argmax(1), average='macro')

	p_logits = F.log_softmax(output['p'], dim=1)
	idx_test_p = label['p'][3]
	x_test_p = p_logits[idx_test_p]
	y_test_p = label['p'][0][idx_test_p]
	f1_micro_test_p = f1_score(y_test_p.data.cpu(), x_test_p.data.cpu().argmax(1), average='micro')
	f1_macro_test_p = f1_score(y_test_p.data.cpu(), x_test_p.data.cpu().argmax(1), average='macro')

	# a_logits = F.log_softmax(output['a'], dim=1)
	# idx_test_a = label['a'][3]
	# x_test_a = a_logits[idx_test_a]
	# y_test_a = label['a'][0][idx_test_a]
	# f1_micro_test_a = f1_score(y_test_a.data.cpu(), x_test_a.data.cpu().argmax(1), average='micro')
	# f1_macro_test_a = f1_score(y_test_a.data.cpu(), x_test_a.data.cpu().argmax(1), average='macro')

	# c_logits = F.log_softmax(output['c'], dim=1)
	# idx_test_c = label['c'][3]
	# x_test_c = c_logits[idx_test_c]
	# y_test_c = label['c'][0][idx_test_c]
	# f1_micro_test_c = f1_score(y_test_c.data.cpu(), x_test_c.data.cpu().argmax(1), average='micro')
	# f1_macro_test_c = f1_score(y_test_c.data.cpu(), x_test_c.data.cpu().argmax(1), average='macro')
	
	print(
		  '\n'
  		  'test micro f1 p: {:.4f}'.format(f1_micro_test_p.item()),
		  'test macro f1 p: {:.4f}'.format(f1_macro_test_p.item()),
  		#   'test micro f1 m: {:.4f}'.format(f1_micro_test_m.item()),
		  # 'test macro f1 m: {:.4f}'.format(f1_macro_test_m.item()),
  		  # 'test micro f1 a: {:.4f}'.format(f1_micro_test_a.item()),
		  # 'test macro f1 a: {:.4f}'.format(f1_macro_test_a.item()),
		 )




if __name__ == '__main__':
	
	cuda = True # Enables CUDA training.
	lr = 0.01 # Initial learning rate.
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64
	gcn = True  # True, False
	
	layer_fusion = 'none'  # none cat att
	type_fusion = 'att'  # mean att
	train_percent = 0.6

	run_num = 40
	for run in range(run_num):
		print('\nHGCN run: ', run, '\n')

		# if run >= 0 and run < 40:
		# 	layer_fusion = 'none'
		# 	type_fusion = 'mean'
		# elif run >= 40 and run < 80:
		# 	layer_fusion = 'att'
		# 	type_fusion = 'att'
		# elif run >= 80 and run < 120:
		# 	layer_fusion = 'none'
		# 	type_fusion = 'att'

		train_percent =+ (int((run % 40) / 10) + 1) * 2 / 10
		
		seed = run % 10

		np.random.seed(seed)
		torch.manual_seed(seed)
		if cuda and torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)


		# hid_layer_dim = [128,64,32]  # imdb3228
		# # hid_layer_dim = [64]  # imdb3228
		# label, ft_dict, adj_dict = load_imdb3228(train_percent)
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)
		# epochs = 250


		hid_layer_dim = [64,32,16] # acm
		# hid_layer_dim = [64]
		epochs = 200
		label, ft_dict, adj_dict = load_acm4025(train_percent)
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 3)


		# hid_layer_dim = [64,32,16]  # dblp4area
		# # hid_layer_dim = [64]
		# label, ft_dict, adj_dict = load_dblp4area(train_percent)
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)
		# epochs = 200

		# hid_layer_dim = [32]  # imdb128
		# label, ft_dict, adj_dict  = load_imdb128()
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 17)

		# hid_layer_dim = [128,64,32]  # imdb10197
		# # hid_layer_dim = [128]
		# label, ft_dict, adj_dict = load_imdb10197()
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 19)

		# hid_layer_dim = [64,32,16] #dbis
		# # hid_layer_dim = [128,128,64]
		# # hid_layer_dim = [128,128,128]
		# # hid_layer_dim = [64,32]
		# # hid_layer_dim = [64]
		# label, ft_dict, adj_dict = load_dbis()
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 8)

		# hid_layer_dim = [64,32,16] # douban
		# # hid_layer_dim = [64] # douban
		# label, ft_dict, adj_dict = load_douban1594()
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)



		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)

		print(
   			 '\n',
			 'layer_fusion: {}\n'.format(layer_fusion), 
			 'type_fusion: {}\n'.format(type_fusion),
			 '\n'
			 )

		# Model and optimizer
		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
		model = HGCN(
					net_schema=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()),
					layer_fusion=layer_fusion,
					type_fusion=type_fusion,
					type_att_size=type_att_size,
					gcn=gcn,
					)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


		if cuda and torch.cuda.is_available():
			model.cuda()

			for k in ft_dict:
				ft_dict[k] = ft_dict[k].cuda()
			for k in adj_dict:
				for kk in adj_dict[k]:
					adj_dict[k][kk] = adj_dict[k][kk].cuda()
			for k in label:
				for i in range(len(label[k])):
					label[k][i] = label[k][i].cuda()

		for epoch in range(epochs):
			train(epoch)
		train(epochs)
		test()