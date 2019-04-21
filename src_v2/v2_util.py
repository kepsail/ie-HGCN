import torch
import numpy as np



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