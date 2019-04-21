import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
import torch

from v2_util import row_normalize, sp_coo_2_sp_tensor



# seed = 87
# np.random.seed(seed)
# torch.manual_seed(seed)



def load_imdb128():
	path='./data/imdb128/'
	dataset='imdb128'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_movie_feature.pkl'.format(path, dataset), 'rb') as in_file:
		m_ft = pickle.load(in_file)

	with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
		(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)


	# label: genre
	label = {}
	m_label = sp_A_m_g.todense()
	m_label = torch.LongTensor(m_label)
	idx_train_m = torch.LongTensor(np.arange(0, int(m_label.shape[0]*1.0)))
	idx_val_m = torch.LongTensor(np.arange(int(m_label.shape[0]*0.8), int(m_label.shape[0]*0.9)))
	idx_test_m = torch.LongTensor(np.arange(int(m_label.shape[0]*0.9), m_label.shape[0]))
	label['m'] = [m_label, idx_train_m, idx_val_m, idx_test_m]


	# feature: movie feature is loaded, other features are genreted by their one-hot vectors
	ft_dict = {}
	m_ft_std = (m_ft - m_ft.mean(0)) / m_ft.std(0)
	ft_dict['m'] = torch.FloatTensor(m_ft_std)
	
	ft_dict['a'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_a.shape[1]))))
	ft_dict['u'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_u.shape[1]))))
	ft_dict['t'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_t.shape[1]))))
	ft_dict['c'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_c.shape[1]))))
	ft_dict['d'] = torch.FloatTensor(np.mat(np.eye((sp_A_m_d.shape[1]))))


	# dense adj mats
	adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'c':{}, 'd':{}}
	adj_dict['m']['a'] = torch.FloatTensor(row_normalize(sp_A_m_a.todense()))
	adj_dict['m']['u'] = torch.FloatTensor(row_normalize(sp_A_m_u.todense()))
	adj_dict['m']['t'] = torch.FloatTensor(row_normalize(sp_A_m_t.todense()))
	adj_dict['m']['c'] = torch.FloatTensor(row_normalize(sp_A_m_c.todense()))
	adj_dict['m']['d'] = torch.FloatTensor(row_normalize(sp_A_m_d.todense()))
	
	adj_dict['a']['m'] = torch.FloatTensor(row_normalize(sp_A_m_a.todense().transpose()))
	adj_dict['u']['m'] = torch.FloatTensor(row_normalize(sp_A_m_u.todense().transpose()))
	adj_dict['t']['m'] = torch.FloatTensor(row_normalize(sp_A_m_t.todense().transpose()))
	adj_dict['c']['m'] = torch.FloatTensor(row_normalize(sp_A_m_c.todense().transpose()))
	adj_dict['d']['m'] = torch.FloatTensor(row_normalize(sp_A_m_d.todense().transpose()))


	return label, ft_dict, adj_dict



def load_imdb10197():
	path='./data/imdb10197/'
	dataset='imdb10197'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_movie_feature.pkl'.format(path, dataset), 'rb') as in_file:
		m_ft = pickle.load(in_file)

	with open('{}{}_sp_adj_mats.pkl'.format(path, dataset), 'rb') as in_file:
		(sp_A_m_a, sp_A_m_c, sp_A_m_d, sp_A_m_t, sp_A_m_u, sp_A_m_g) = pickle.load(in_file)


	# label: genre
	label = {}
	m_label = sp_A_m_g.todense()
	m_label = np.delete(m_label, -4, 1)


	m_label = torch.LongTensor(m_label)   # multi label indicator
	
	# m_single_label = np.zeros(m_label.shape[0], dtype=np.int64)    # multi class one hot but single label for one example
	# for i in range(m_label.shape[0]):
	# 	(r_idx, c_idx) = np.where(m_label[i] == 1)
	# 	if len(c_idx) == 1:
	# 		m_single_label[i] = c_idx[0]
	# 	else:
	# 		sample_idx = np.random.randint(len(c_idx))
	# 		m_single_label[i] = c_idx[sample_idx]
	# m_label = torch.LongTensor(m_single_label)


	# rand_idx = np.random.permutation(m_label.shape[0])
	# idx_7592 = np.where(rand_idx == 7592)[0]
	# rand_idx = np.delete(rand_idx, idx_7592[0], 0)
	# idx_train_m = torch.LongTensor(rand_idx[int(m_label.shape[0]*0): int(m_label.shape[0]*0.80)])
	# idx_val_m = torch.LongTensor(rand_idx[int(m_label.shape[0]*0.80): int(m_label.shape[0]*0.90)])
	# idx_test_m = torch.LongTensor(rand_idx[int(m_label.shape[0]*0.90): int(m_label.shape[0]*1)])


	idx_train_m = torch.LongTensor(np.arange(int(m_label.shape[0]*0.2), int(m_label.shape[0]*1.0)))
	idx_7592 = np.where(idx_train_m == 7592)[0]
	idx_train_m = np.delete(idx_train_m, idx_7592[0], 0)

	idx_val_m = torch.LongTensor(np.arange(int(m_label.shape[0]*0.0), int(m_label.shape[0]*0.1)))
	idx_test_m = torch.LongTensor(np.arange(int(m_label.shape[0]*0.1), int(m_label.shape[0]*0.2)))



	label['m'] = [m_label, idx_train_m, idx_val_m, idx_test_m]

	# feature: movie feature is loaded, other features are genreted by xavier_uniform distribution
	ft_dict = {}
	m_ft_std = (m_ft - m_ft.mean(0)) / m_ft.std(0)
	ft_dict['m'] = torch.FloatTensor(m_ft_std)
	
	ft_dict['a'] = torch.FloatTensor(sp_A_m_a.shape[1], 256)
	torch.nn.init.xavier_uniform_(ft_dict['a'].data, gain=1.414)
	ft_dict['u'] = torch.FloatTensor(sp_A_m_u.shape[1], 256)
	torch.nn.init.xavier_uniform_(ft_dict['u'].data, gain=1.414)
	# ft_dict['t'] = torch.FloatTensor(sp_A_m_t.shape[1], 256)
	# torch.nn.init.xavier_uniform_(ft_dict['t'].data, gain=1.414)
	# ft_dict['c'] = torch.FloatTensor(sp_A_m_c.shape[1], 256)
	# torch.nn.init.xavier_uniform_(ft_dict['c'].data, gain=1.414)
	ft_dict['d'] = torch.FloatTensor(sp_A_m_d.shape[1], 256)
	torch.nn.init.xavier_uniform_(ft_dict['d'].data, gain=1.414)
	

	# sparse adj mats
	# adj_dict = {'m':{}, 'a':{}, 'u':{}, 't':{}, 'c':{}, 'd':{}}
	adj_dict = {'m':{}, 'a':{}, 'u':{}, 'd':{}}
	adj_dict['m']['a'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_a.todense())))
	adj_dict['m']['u'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_u.todense())))
	# adj_dict['m']['t'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_t.todense())))
	# adj_dict['m']['c'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_c.todense())))
	adj_dict['m']['d'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_d.todense())))
	
	adj_dict['a']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_a.todense().transpose())))
	adj_dict['u']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_u.todense().transpose())))
	# adj_dict['t']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_t.todense().transpose())))
	# adj_dict['c']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_c.todense().transpose())))
	adj_dict['d']['m'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_m_d.todense().transpose())))


	# dataset split mask
	return label, ft_dict, adj_dict



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
	label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]

	label_index_a = np.where(a_label != -1)[0]
	a_label = torch.LongTensor(a_label)
	idx_train_a = torch.LongTensor(label_index_a[0: int(len(label_index_a)*0.8)])
	idx_val_a = torch.LongTensor(label_index_a[int(len(label_index_a)*0.8): int(len(label_index_a)*0.9)])
	idx_test_a = torch.LongTensor(label_index_a[int(len(label_index_a)*0.9): ])
	label['a'] = [a_label, idx_train_a, idx_val_a, idx_test_a]


	label_index_c = np.where(c_label != -1)[0]
	c_label = torch.LongTensor(c_label)
	idx_train_c = torch.LongTensor(label_index_c[0: int(len(label_index_c)*0.8)])
	idx_val_c = torch.LongTensor(label_index_c[int(len(label_index_c)*0.8): int(len(label_index_c)*0.9)])
	idx_test_c = torch.LongTensor(label_index_c[int(len(label_index_c)*0.9): ])
	label['c'] = [c_label, idx_train_c, idx_val_c, idx_test_c]


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
	

	adj_dict = {'p':{}, 'a':{}, 'c':{}, 't':{}}
	
	# sparse adj mats
	adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense())))
	adj_dict['p']['c'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense())))
	adj_dict['p']['t'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_t.todense())))
	
	adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense().transpose())))
	adj_dict['c']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense().transpose())))
	adj_dict['t']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_t.todense().transpose())))

	# dense adj mats
	# adj_dict['p']['a'] = torch.FloatTensor(row_normalize(sp_A_p_a.todense()))
	# adj_dict['p']['c'] = torch.FloatTensor(row_normalize(sp_A_p_c.todense()))
	# adj_dict['p']['t'] = torch.FloatTensor(row_normalize(sp_A_p_t.todense()))
	
	# adj_dict['a']['p'] = torch.FloatTensor(row_normalize(sp_A_p_a.todense().transpose()))
	# adj_dict['c']['p'] = torch.FloatTensor(row_normalize(sp_A_p_c.todense().transpose()))
	# adj_dict['t']['p'] = torch.FloatTensor(row_normalize(sp_A_p_t.todense().transpose()))


	return label, ft_dict, adj_dict



def load_dbis():
	path='./data/dbis/'
	dataset='dbis'
	print('Loading {} dataset...'.format(dataset))

	with open('{}{}_paper_feature.pkl'.format(path, dataset), 'rb') as in_file:
		p_ft = pickle.load(in_file)

	with open('{}{}_author_label.pkl'.format(path, dataset), 'rb') as in_file:
		author_label = np.array(pickle.load(in_file), dtype=np.int64)
		author_label[:,1] = author_label[:,1] - 1

	norm_adj_mats = sio.loadmat('{}{}_sp_row_norm_adj_mats.mat'.format(path, dataset))


	# author label
	label = {}

	a_label = np.full(norm_adj_mats['norm_p_a'].shape[1], -1, dtype=np.int64)
	a_label[author_label[:,0]] = author_label[:,1]
	a_label = torch.LongTensor(a_label)

	# idx_train_a = torch.LongTensor(author_label[:,0][int(author_label.shape[0]*0): int(author_label.shape[0]*0.8)])
	# idx_val_a = torch.LongTensor(author_label[:,0][int(author_label.shape[0]*0.8): int(author_label.shape[0]*0.9)])
	# idx_test_a = torch.LongTensor(author_label[:,0][int(author_label.shape[0]*0.9): int(author_label.shape[0]*1)])

	idx_train_a = []
	np.random.shuffle(author_label)

	# author_label_train = author_label[int(author_label.shape[0]*0.0):int(author_label.shape[0]*0.8)]
	# cate_num = np.unique(author_label_train[:,1]).shape[0]
	# if cate_num == 8:
	# 	max_cate_num = np.bincount(author_label_train[:,1]).max()
	# 	for cate_i in range(cate_num):
	# 		cate_i_idx = author_label_train[np.where(author_label_train[:,1]==cate_i)[0], 0]
	# 		idx_train_a.extend(cate_i_idx[:max_cate_num])
	# 	np.random.shuffle(idx_train_a)
	# 	idx_train_a = torch.LongTensor(idx_train_a)
	# else:
	# 	print('please check the train label distribution!')
	# 	exit()

	idx_train_a = torch.LongTensor(author_label[int(author_label.shape[0]*0.0):int(author_label.shape[0]*0.8), 0])
	idx_val_a = torch.LongTensor(author_label[int(author_label.shape[0]*0.8):int(author_label.shape[0]*0.9), 0])
	idx_test_a = torch.LongTensor(author_label[int(author_label.shape[0]*0.9):int(author_label.shape[0]*1.0), 0])




	label['a'] = [a_label, idx_train_a, idx_val_a, idx_test_a]
	
	# feature: paper feature is loaded, other features are genreted by xavier_uniform distribution
	ft_dict = {}
	p_ft_std = (p_ft - p_ft.mean(0)) / p_ft.std(0)
	ft_dict['p'] = torch.FloatTensor(p_ft_std)
	
		
	ft_dict['a'] = torch.FloatTensor(norm_adj_mats['norm_p_a'].shape[1], 128)
	torch.nn.init.xavier_uniform_(ft_dict['a'].data, gain=1.414)
	ft_dict['c'] = torch.FloatTensor(norm_adj_mats['norm_p_c'].shape[1], 128)
	torch.nn.init.xavier_uniform_(ft_dict['c'].data, gain=1.414)
	

	# sparse adj mats
	adj_dict = {'p':{}, 'a':{}, 'c':{}}

	# adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense())))
	# adj_dict['p']['c'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense())))
	
	# adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_a.todense().transpose())))
	# adj_dict['c']['p'] = sp_coo_2_sp_tensor(sp.coo_matrix(row_normalize(sp_A_p_c.todense().transpose())))

	adj_dict['p']['a'] = sp_coo_2_sp_tensor(norm_adj_mats['norm_p_a'].astype(np.float32).tocoo())
	adj_dict['p']['c'] = sp_coo_2_sp_tensor(norm_adj_mats['norm_p_c'].astype(np.float32).tocoo())
	
	adj_dict['a']['p'] = sp_coo_2_sp_tensor(norm_adj_mats['norm_a_p'].astype(np.float32).tocoo())
	adj_dict['c']['p'] = sp_coo_2_sp_tensor(norm_adj_mats['norm_c_p'].astype(np.float32).tocoo())


	return label, ft_dict, adj_dict


if __name__ == '__main__':
	load_imdb128()
	# load_imdb10197()
	# load_dblp4area()
	# load_dbis()