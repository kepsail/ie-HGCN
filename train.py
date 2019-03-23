import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from util import *
from model import HGCN



cuda = False # Enables CUDA training.
fastmode = True # Validate during training pass.
seed = 87 # Random seed.
epochs = 100 # Number of epochs to train.P
lr = 0.01 # Initial learning rate.
weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
# eq_hidden = [512, 128, 16] # imdb10197
eq_hidden = [64, 32, 16] # dblp4area
dropout = 0.5 # Dropout rate (1 - keep probability).


# Set seeds
np.random.seed(seed)
torch.manual_seed(seed)
if cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load data
# label, ft_dict, adj_dict, idx_train, idx_val, idx_test = load_imdb128()
# label, ft_dict, adj_dict, idx_train, idx_val, idx_test = load_imdb10197()
label, ft_dict, adj_dict = load_dblp4area()


layer_shape = []
input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
layer_shape.append(input_layer_shape)
hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in eq_hidden]
layer_shape.extend(hidden_layer_shape)
output_layer_shape = dict.fromkeys(ft_dict.keys(), np.unique(label[list(label.keys())[0]][0]).shape[0] - 1)
for k in label.keys():
	output_layer_shape[k] = np.unique(label[k][0]).shape[0] - 1
layer_shape.append(output_layer_shape)


# Model and optimizer
net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
model = HGCN(net_schema=net_schema, layer_shape=layer_shape, dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



def train(epochs):

	model.train()
	optimizer.zero_grad()
	output = model(ft_dict, adj_dict)
	

	idx_train = label['a'][1]
	x_train = F.log_softmax(output['a'], dim=1)[idx_train]
	y_train = label['a'][0][idx_train]
	loss_train = F.nll_loss(x_train, y_train)
	acc_train = accuracy(x_train, y_train)
	print('a', 'loss: {:.4f}'.format(loss_train.item()), 'acc: {:.4f}'.format(acc_train.item()))

	
	# x_train = F.log_softmax(output[label[0]], dim=1)[idx_train]
	# y_train = torch.nonzero(label[1][idx_train])[:,1]
	# loss_train = F.nll_loss(x_train, y_train)
	# acc_train = accuracy(x_train, y_train)
	# print('a', 'loss: {:.4f}'.format(loss_train.item()), 'acc: {:.4f}'.format(acc_train.item()))


	# x_train = output[label[0]][idx_train]
	# y_train = torch.nonzero(label[1][idx_train])[:,1]
	# loss_train = F.cross_entropy(x_train, y_train)
	# acc_train = accuracy(x_train, y_train)
	# print('a', 'loss: {:.4f}'.format(loss_train.item()), 'acc: {:.4f}'.format(acc_train.item()))


	# x_train = torch.sigmoid(output[label[0]])[idx_train]
	# y_train = label[1][idx_train]
	# loss_train = F.binary_cross_entropy(x_train, y_train)
	# acc_train = roc_auc_score(y_train.detach().numpy(), x_train.detach().numpy())

	
	loss_train.backward()
	optimizer.step()


if __name__ == '__main__':

	for epoch in range(epochs):
		train(epoch)