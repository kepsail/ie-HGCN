import numpy as np
import scipy.sparse as sp
import pickle
import torch



def load_imdb128():
	path='./data/imdb128/'
	dataset='imdb128'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_movie_feature.pkl'.format(path, dataset), 'rb') as in_file:
		m_ft = pickle.load(in_file)

	with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
		(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)


	# label: country or genre
	# label = ('m', torch.LongTensor(sp_A_m_g.todense()))
	label = ('m', torch.LongTensor(sp_A_m_c.todense()))


	# feature: movie feature is loaded, other features are genreted by their one-hot vectors
	ft_dict = {}
	m_ft_std = (m_ft - m_ft.mean(0)) / m_ft.std(0)
	ft_dict['m'] = torch.FloatTensor(m_ft_std)
	
	ft_dict['a'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_a.shape[1]))))
	ft_dict['u'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_u.shape[1]))))
	ft_dict['t'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_t.shape[1]))))
	# ft_dict['c'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_c.shape[1]))))
	ft_dict['d'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_d.shape[1]))))


	# dense adj mats
	# adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'c':{}, 'd':{}}
	adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'd':{}}
	adj_dict['m']['a'] = torch.FloatTensor(row_normalize(sp_A_m_a.todense()))
	adj_dict['m']['u'] = torch.FloatTensor(row_normalize(sp_A_m_u.todense()))
	adj_dict['m']['t'] = torch.FloatTensor(row_normalize(sp_A_m_t.todense()))
	# adj_dict['m']['c'] = torch.FloatTensor(row_normalize(sp_A_m_c.todense()))
	adj_dict['m']['d'] = torch.FloatTensor(row_normalize(sp_A_m_d.todense()))
	
	adj_dict['a']['m'] = torch.FloatTensor(row_normalize(sp_A_m_a.todense().transpose()))
	adj_dict['u']['m'] = torch.FloatTensor(row_normalize(sp_A_m_u.todense().transpose()))
	adj_dict['t']['m'] = torch.FloatTensor(row_normalize(sp_A_m_t.todense().transpose()))
	# adj_dict['c']['m'] = torch.FloatTensor(row_normalize(sp_A_m_c.todense().transpose()))
	adj_dict['d']['m'] = torch.FloatTensor(row_normalize(sp_A_m_d.todense().transpose()))


	# dataset split mask
	idx_train = torch.LongTensor(range(0, 60))
	idx_val = torch.LongTensor(range(60, 100))
	idx_test = torch.LongTensor(range(100, 128))


	return label, ft_dict, adj_dict, idx_train, idx_val, idx_test



def load_imdb10197():
	path='./data/imdb10197/'
	dataset='imdb10197'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_movie_feature.pkl'.format(path, dataset), 'rb') as in_file:
		m_ft = pickle.load(in_file)

	with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
		(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)


	# label: country or genre
	# label = ('m', torch.LongTensor(sp_A_m_g.todense()))
	label = ('m', torch.LongTensor(sp_A_m_c.todense()))


	# feature: movie feature is loaded, other features are genreted by xavier_uniform distribution
	ft_dict = {}
	m_ft_std = (m_ft - m_ft.mean(0)) / m_ft.std(0)
	ft_dict['m'] = torch.FloatTensor(m_ft_std)
	
	ft_dict['a'] = torch.FloatTensor(sp_A_m_a.shape[1], 2**8)
	torch.nn.init.xavier_uniform_(ft_dict['a'].data, gain=1.414)
	ft_dict['u'] = torch.FloatTensor(sp_A_m_u.shape[1], 2**8)
	torch.nn.init.xavier_uniform_(ft_dict['u'].data, gain=1.414)
	ft_dict['t'] = torch.FloatTensor(sp_A_m_t.shape[1], 2**8)
	torch.nn.init.xavier_uniform_(ft_dict['t'].data, gain=1.414)
	# ft_dict['c'] = torch.FloatTensor(sp_A_m_c.shape[1], 2**8)
	# torch.nn.init.xavier_uniform_(ft_dict['c'].data, gain=1.414)
	ft_dict['d'] = torch.FloatTensor(sp_A_m_d.shape[1], 2**8)
	torch.nn.init.xavier_uniform_(ft_dict['d'].data, gain=1.414)
	

	# sparse adj mats
	# adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'c':{}, 'd':{}}
	adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'd':{}}
	adj_dict['m']['a'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_a.todense())))
	adj_dict['m']['u'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_u.todense())))
	adj_dict['m']['t'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_t.todense())))
	# adj_dict['m']['c'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_c.todense())))
	adj_dict['m']['d'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_d.todense())))
	
	adj_dict['a']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_a.todense().transpose())))
	adj_dict['u']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_u.todense().transpose())))
	adj_dict['t']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_t.todense().transpose())))
	# adj_dict['c']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_c.todense().transpose())))
	adj_dict['d']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_d.todense().transpose())))


	# dataset split mask
	idx_train = torch.LongTensor(range(0, 8000))
	idx_val = torch.LongTensor(range(8000, 10000))
	idx_test = torch.LongTensor(range(10000, 10197))


	return label, ft_dict, adj_dict, idx_train, idx_val, idx_test



def load_dblp4area():
	path='./data/dblp4area/'
	dataset='dblp4area'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_paper_feature.pkl'.format(path, dataset), 'rb') as in_file:
		p_ft = pickle.load(in_file)

	with open('{}{}_label.pkl'.format(path, dataset), 'rb') as in_file:
		(p_label, a_label, c_label) = pickle.load(in_file)

	with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
		(sp_A_p_a, sp_A_p_c, sp_A_p_t) = pickle.load(in_file)


	# label and dataset split idx
	label = {}
	
	label_index_p = np.where(p_label != -1)[0]
	p_label = torch.LongTensor(p_label)
	idx_train_p = torch.LongTensor(label_index_p[0: int(len(label_index_p)*0.8)])
	idx_val_p = torch.LongTensor(label_index_p[int(len(label_index_p)*0.8): int(len(label_index_p)*0.9)])
	idx_test_p = torch.LongTensor(label_index_p[int(len(label_index_p)*0.9): ])
	label['p'] = (p_label, idx_train_p, idx_val_p, idx_test_p)

	label_index_a = np.where(a_label != -1)[0]
	a_label = torch.LongTensor(a_label)
	idx_train_a = torch.LongTensor(label_index_a[0: int(len(label_index_a)*0.8)])
	idx_val_a = torch.LongTensor(label_index_a[int(len(label_index_a)*0.8): int(len(label_index_a)*0.9)])
	idx_test_a = torch.LongTensor(label_index_a[int(len(label_index_a)*0.9): ])
	label['a'] = (a_label, idx_train_a, idx_val_a, idx_test_a)


	label_index_c = np.where(c_label != -1)[0]
	c_label = torch.LongTensor(c_label)
	idx_train_c = torch.LongTensor(label_index_c[0: int(len(label_index_c)*0.8)])
	idx_val_c = torch.LongTensor(label_index_c[int(len(label_index_c)*0.8): int(len(label_index_c)*0.9)])
	idx_test_c = torch.LongTensor(label_index_c[int(len(label_index_c)*0.9): ])
	label['c'] = (c_label, idx_train_c, idx_val_c, idx_test_c)


	# feature: paper feature is loaded, other features are genreted by xavier_uniform distribution
	ft_dict = {}
	p_ft_std = (p_ft - p_ft.mean(0)) / p_ft.std(0)
	ft_dict['p'] = torch.FloatTensor(p_ft_std)
	
	ft_dict['a'] = torch.FloatTensor(sp_A_p_a.shape[1], 2**7)
	torch.nn.init.xavier_uniform_(ft_dict['a'].data, gain=1.414)
	ft_dict['c'] = torch.FloatTensor(sp_A_p_c.shape[1], 2**7)
	torch.nn.init.xavier_uniform_(ft_dict['c'].data, gain=1.414)
	ft_dict['t'] = torch.FloatTensor(sp_A_p_t.shape[1], 2**7)
	torch.nn.init.xavier_uniform_(ft_dict['t'].data, gain=1.414)
	

	# sparse adj mats
	adj_dict = {'p':{}, 'a':{}, 'c':{}, 't':{}}
	adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense())))
	adj_dict['p']['c'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense())))
	adj_dict['p']['t'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_t.todense())))
	
	adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense().transpose())))
	adj_dict['c']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense().transpose())))
	adj_dict['t']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_t.todense().transpose())))



	return label, ft_dict, adj_dict



def row_normalize(mat):
    """Row-normalize matrix"""
    rowsum = mat.sum(1)
    rowsum[rowsum == 0.] = 0.01
    return mat / rowsum



def sp_coo_2_sp_tensor(sp_coo_mat):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


if __name__ == '__main__':
	# load_imdb10197()
	load_dblp4area()