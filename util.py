import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer



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



def load_imdb3228(train_percent):
	hgcn_path = './data/imdb3228/imdb3228_hgcn_'+str(train_percent)+'.pkl'
	print('hgcn load: ', hgcn_path, '\n')
	with open(hgcn_path, 'rb') as in_file:
		(label, ft_dict, adj_dict) = pickle.load(in_file)
		adj_dict['m']['a'] = adj_dict['m']['a'].to_sparse()
		adj_dict['m']['u'] = adj_dict['m']['u'].to_sparse()
		adj_dict['m']['d'] = adj_dict['m']['d'].to_sparse()
		
		adj_dict['a']['m'] = adj_dict['a']['m'].to_sparse()
		adj_dict['u']['m'] = adj_dict['u']['m'].to_sparse()
		adj_dict['d']['m'] = adj_dict['d']['m'].to_sparse()

	return label, ft_dict, adj_dict



def load_acm4025(train_percent):
	hgcn_path = './data/acm4025/acm4025_hgcn_'+str(train_percent)+'.pkl'
	print('hgcn load: ', hgcn_path, '\n')
	with open(hgcn_path, 'rb') as in_file:
		(label, ft_dict, adj_dict) = pickle.load(in_file)

		adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
		adj_dict['p']['l'] = adj_dict['p']['l'].to_sparse()
		
		adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
		adj_dict['l']['p'] = adj_dict['l']['p'].to_sparse()
	
	return label, ft_dict, adj_dict



def load_dblp4area4057(train_percent):
	hgcn_path = './data/dblp4area4057/dblp4area4057_hgcn_'+str(train_percent)+'.pkl'
	print('hgcn load: ', hgcn_path, '\n')
	with open(hgcn_path, 'rb') as in_file:
		(label, ft_dict, adj_dict) = pickle.load(in_file)
	
		adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
		adj_dict['p']['c'] = adj_dict['p']['c'].to_sparse()
		adj_dict['p']['t'] = adj_dict['p']['t'].to_sparse()
		
		adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
		adj_dict['c']['p'] = adj_dict['c']['p'].to_sparse()
		adj_dict['t']['p'] = adj_dict['t']['p'].to_sparse()

	return label, ft_dict, adj_dict



if __name__ == '__main__':
	load_imdb3228(0.2)	
	# load_imdb3228(0.4)	
	# load_imdb3228(0.6)	
	# load_imdb3228(0.8)	
	# load_acm4025(0.2)
	# load_acm4025(0.4)
	# load_acm4025(0.6)
	# load_acm4025(0.8)
	# load_dblp4area4057(0.2)
	# load_dblp4area4057(0.4)
	# load_dblp4area4057(0.6)
	# load_dblp4area4057(0.8)
