import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import time

from util import *
from model import HGCN



def train(epoch):

	model.train()
	optimizer.zero_grad()
	logits, _ = model(ft_dict, adj_dict)



	# m_logits = F.log_softmax(logits['m'], dim=1)
	# idx_train_m = label['m'][1]
	# x_train_m = m_logits[idx_train_m]
	# y_train_m = label['m'][0][idx_train_m]
	# loss_train = F.nll_loss(x_train_m, y_train_m)
	# f1_micro_train_m = f1_score(y_train_m.data.cpu(), x_train_m.data.cpu().argmax(1), average='micro')
	# f1_macro_train_m = f1_score(y_train_m.data.cpu(), x_train_m.data.cpu().argmax(1), average='macro')

	# p_logits = F.log_softmax(logits['p'], dim=1)
	# idx_train_p = label['p'][1]
	# x_train_p = p_logits[idx_train_p]
	# y_train_p = label['p'][0][idx_train_p]
	# loss_train = F.nll_loss(x_train_p, y_train_p)
	# f1_micro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='micro')
	# f1_macro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='macro')

	a_logits = F.log_softmax(logits['a'], dim=1)
	idx_train_a = label['a'][1]
	x_train_a = a_logits[idx_train_a]
	y_train_a = label['a'][0][idx_train_a]
	loss_train = F.nll_loss(x_train_a, y_train_a)
	f1_micro_train_a = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='micro')
	f1_macro_train_a = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='macro')

	loss_train.backward()
	optimizer.step()



	'''///////////////// Validatin ///////////////////'''
	model.eval()
	logits, _ = model(ft_dict, adj_dict)



	# m_logits = F.log_softmax(logits['m'], dim=1)
	# idx_val_m = label['m'][2]
	# x_val_m = m_logits[idx_val_m]
	# y_val_m = label['m'][0][idx_val_m]
	# f1_micro_val_m = f1_score(y_val_m.data.cpu(), x_val_m.data.cpu().argmax(1), average='micro')
	# f1_macro_val_m = f1_score(y_val_m.data.cpu(), x_val_m.data.cpu().argmax(1), average='macro')

	# p_logits = F.log_softmax(logits['p'], dim=1)
	# idx_val_p = label['p'][2]
	# x_val_p = p_logits[idx_val_p]
	# y_val_p = label['p'][0][idx_val_p]
	# f1_micro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='micro')
	# f1_macro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='macro')

	a_logits = F.log_softmax(logits['a'], dim=1)
	idx_val_a = label['a'][2]
	x_val_a = a_logits[idx_val_a]
	y_val_a = label['a'][0][idx_val_a]
	f1_micro_val_a = f1_score(y_val_a.data.cpu(), x_val_a.data.cpu().argmax(1), average='micro')
	f1_macro_val_a = f1_score(y_val_a.data.cpu(), x_val_a.data.cpu().argmax(1), average='macro')

	
	if epoch % 1 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  # 'train micro f1 m: {:.4f}'.format(f1_micro_train_m.item()),
			  # 'train macro f1 m: {:.4f}'.format(f1_macro_train_m.item()),
			  # 'val micro f1 m: {:.4f}'.format(f1_micro_val_m.item()),
			  # 'val macro f1 m: {:.4f}'.format(f1_macro_val_m.item()),
			  # 'train micro f1 p: {:.4f}'.format(f1_micro_train_p.item()),
			  # 'train macro f1 p: {:.4f}'.format(f1_macro_train_p.item()),
			  # 'val micro f1 p: {:.4f}'.format(f1_micro_val_p.item()),
			  # 'val macro f1 p: {:.4f}'.format(f1_macro_val_p.item()),
			  'train micro f1 a: {:.4f}'.format(f1_micro_train_a.item()),
			  'train macro f1 a: {:.4f}'.format(f1_macro_train_a.item()),
			  'val micro f1 a: {:.4f}'.format(f1_micro_val_a.item()),
			  'val macro f1 a: {:.4f}'.format(f1_macro_val_a.item()),
			 )



def test():
	model.eval()
	logits, embd = model(ft_dict, adj_dict)



	# m_logits = F.log_softmax(logits['m'], dim=1)
	# idx_test_m = label['m'][3]
	# x_test_m = m_logits[idx_test_m]
	# y_test_m = label['m'][0][idx_test_m]
	# f1_micro_test_m = f1_score(y_test_m.data.cpu(), x_test_m.data.cpu().argmax(1), average='micro')
	# f1_macro_test_m = f1_score(y_test_m.data.cpu(), x_test_m.data.cpu().argmax(1), average='macro')

	# p_logits = F.log_softmax(logits['p'], dim=1)
	# idx_test_p = label['p'][3]
	# x_test_p = p_logits[idx_test_p]
	# y_test_p = label['p'][0][idx_test_p]
	# f1_micro_test_p = f1_score(y_test_p.data.cpu(), x_test_p.data.cpu().argmax(1), average='micro')
	# f1_macro_test_p = f1_score(y_test_p.data.cpu(), x_test_p.data.cpu().argmax(1), average='macro')

	a_logits = F.log_softmax(logits['a'], dim=1)
	idx_test_a = label['a'][3]
	x_test_a = a_logits[idx_test_a]
	y_test_a = label['a'][0][idx_test_a]
	f1_micro_test_a = f1_score(y_test_a.data.cpu(), x_test_a.data.cpu().argmax(1), average='micro')
	f1_macro_test_a = f1_score(y_test_a.data.cpu(), x_test_a.data.cpu().argmax(1), average='macro')

	
	print(
		  '\n'+
  		  # 'test micro f1 m: {:.4f}'.format(f1_micro_test_m.item()),
		  # 'test macro f1 m: {:.4f}'.format(f1_macro_test_m.item()),
  		  # 'test micro f1 p: {:.4f}'.format(f1_micro_test_p.item()),
		  # 'test macro f1 p: {:.4f}'.format(f1_macro_test_p.item()),
  		  'test micro f1 a: {:.4f}'.format(f1_micro_test_a.item()),
		  'test macro f1 a: {:.4f}'.format(f1_macro_test_a.item()),
		 )



	# return (f1_micro_test_m, f1_macro_test_m)
	# return (f1_micro_test_p, f1_macro_test_p)
	return (f1_micro_test_a, f1_macro_test_a)



if __name__ == '__main__':

	cuda = True # Enables CUDA training.
	lr = 0.01 # Initial learning rate.c
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean
	

	# # scalability
	# author_num = [14475,10000,7000,5500,4000,2500,1500,800]
	# epochs = 500

	# # depth
	# hid_layer_dim = [64]
	# epochs = 200
	# hid_layer_dim = [64,32]
	# epochs = 200
	# hid_layer_dim = [64,32,16]
	# epochs = 200
	# hid_layer_dim = [64,32,16,8]
	# epochs = 200
	# hid_layer_dim = [64,64,32,16,8]
	# epochs = 200
	# hid_layer_dim = [64,64,64,32,16,8]
	# epochs = 200
	# hid_layer_dim = [64,64,64,64,32,16,8]
	# epochs = 350
	# hid_layer_dim = [64,64,64,64,64,32,16,8]
	# epochs = 650
	# hid_layer_dim = [64,64,64,64,64,64,32,16,8]
	# epochs = 3000


	run_num = 1
	train_percent = 0.2
	for run in range(run_num):
		t_start = time.time()
		seed = run

		np.random.seed(seed)
		torch.manual_seed(seed)
		if cuda and torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)

		print('\nHGCN run: ', run)
		print('train percent: ', train_percent)
		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)


		# hid_layer_dim = [64,32,16,8]  # imdb3228
		# epochs = 250
		# label, ft_dict, adj_dict = load_imdb3228(train_percent)
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)

		# hid_layer_dim = [64,32,16,8] # acm
		# epochs = 130
		# label, ft_dict, adj_dict = load_acm4025(train_percent)
		# output_layer_shape = dict.fromkeys(ft_dict.keys(), 3)

		hid_layer_dim = [64,32,16,8]  # dblp4area4057
		epochs = 200
		label, ft_dict, adj_dict = load_dblp4area4057(train_percent)
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)


		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)


		# Model and optimizer
		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
		model = HGCN(
					net_schema=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()),
					type_fusion=type_fusion,
					type_att_size=type_att_size,
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

		(micro_f1, macro_f1) = test()

		t_end = time.time()
		print('Total time: ', t_end - t_start)