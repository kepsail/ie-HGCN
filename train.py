import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from util import *
from model import HGCN



cuda = True # Enables CUDA training.
fastmode = True # Validate during training pass.
seed = 87 # Random seed.
epochs = 1000 # Number of epochs to train.P
lr = 0.01 # Initial learning rate.
weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
dropout = 0.5 # Dropout rate (1 - keep probability).


# Set seeds
np.random.seed(seed)
torch.manual_seed(seed)
if cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# eq_hidden = [32] # imdb128
# label, ft_dict, adj_dict  = load_imdb128()
# output_layer_shape = dict.fromkeys(ft_dict.keys(), 17)


eq_hidden = [128,64,32] # imdb10197
label, ft_dict, adj_dict = load_imdb10197()
output_layer_shape = dict.fromkeys(ft_dict.keys(), 20)

# eq_hidden = [64,16,8] # dblp4area
# label, ft_dict, adj_dict = load_dblp4area()
# output_layer_shape = dict.fromkeys(ft_dict.keys(), 4)


layer_shape = []
input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
layer_shape.append(input_layer_shape)
hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in eq_hidden]
layer_shape.extend(hidden_layer_shape)
layer_shape.append(output_layer_shape)


# Model and optimizer
net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
model = HGCN(net_schema=net_schema, layer_shape=layer_shape, dropout=dropout)
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


def train(epochs):

	model.train()
	optimizer.zero_grad()
	output = model(ft_dict, adj_dict)
	

	# idx_train_p = label['p'][1]
	# x_train_p = F.log_softmax(output['p'], dim=1)[idx_train_p]
	# y_train_p = label['p'][0][idx_train_p]
	# loss_train_p = F.nll_loss(x_train_p, y_train_p)
	# acc_train_p = accuracy(x_train_p, y_train_p)

	# idx_train_a = label['a'][1]
	# x_train_a = F.log_softmax(output['a'], dim=1)[idx_train_a]
	# y_train_a = label['a'][0][idx_train_a]
	# loss_train_a = F.nll_loss(x_train_a, y_train_a)
	# acc_train_a = accuracy(x_train_a, y_train_a)

	# idx_train_c = label['c'][1]
	# x_train_c = F.log_softmax(output['c'], dim=1)[idx_train_c]
	# y_train_c = label['c'][0][idx_train_c]
	# loss_train_c = F.nll_loss(x_train_c, y_train_c)
	# acc_train_c = accuracy(x_train_c, y_train_c)
	
	# loss_train = 0.03*loss_train_p + 0.965*loss_train_a + 0.005*loss_train_c
	# print('epoch: {:3d} loss total: {:.4f} loss_p: {:.4f} loss_a: {:.4f} loss_c: {:.4f} acc_p: {:.4f} acc_a: {:.4f} acc_c: {:.4f}'.format(epochs, loss_train.item(), loss_train_p.item(), loss_train_a.item(), loss_train_c.item(), acc_train_p.item(), acc_train_a.item(), acc_train_c.item()))


	idx_train = label['m'][1]
	x_train = torch.sigmoid(output['m'])[idx_train]
	y_train = label['m'][0][idx_train].type_as(x_train)
	loss_train = F.binary_cross_entropy(x_train, y_train, reduction='none').mean(0).sum()
	roc_train = roc_auc_score(y_train.cpu().detach().numpy(), x_train.cpu().detach().numpy())
	ap_train = average_precision_score(y_train.cpu().detach().numpy(), x_train.cpu().detach().numpy())
	# micro_f1_train = f1_score((y_train>0.5).cpu().detach().numpy(), (x_train>0.5).cpu().detach().numpy(), average='micro')
	# macro_f1_train = f1_score((y_train>0.5).cpu().detach().numpy(), (x_train>0.5).cpu().detach().numpy(), average='macro')
	# print('epoch: {:3d} loss: {:.4f} roc: {:.4f} ap: {:.4f} micro_f1: {:.4f} macro_f1: {:.4f}'.format(epochs, loss_train.item(), roc_train, ap_train, micro_f1_train, macro_f1_train))
	print('epoch: {:3d} loss: {:.4f} roc: {:.4f} ap: {:.4f}'.format(epochs, loss_train.item(), roc_train, ap_train))


	loss_train.backward()
	optimizer.step()


if __name__ == '__main__':

	for epoch in range(epochs):
		train(epoch)
	train(epochs)