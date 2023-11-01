import torch
import math
import random
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from dataset_loader import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def index_to_index(index, size):
    mask = torch.zeros(size, dtype=torch.float)
    mask[index] = 1
    return mask


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    train = index_to_index(train_idx, size=data.num_nodes)
    # train = index_to_index(train_idx, size=data.num_nodes)

    return data,train


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_sparse_tensor(edge_index):
    """Convert edge_index to sparse matrix"""
    sparse_mx = to_scipy_sparse_matrix(edge_index)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.array([sparse_mx.row, sparse_mx.col]))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )


def edge_index_to_adj(edge_index):
    adj = to_sparse_tensor(edge_index)
    adj = adj.to_dense()
    one = torch.ones_like(adj)
    adj = adj + adj.t()  # 对称化
    adj = torch.where(adj < 1, adj, one)
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)  # 去除自环
    adj = adj - a_diag
    # adjaddI = adj + torch.eye(adj.shape[0]) #加自环
    # d1 = torch.sum(adjaddI, dim=1)
    return adj  # 稠密矩阵


#def normalized_laplacian(adj):
#    #adj = sp.coo_matrix(adj)
#    row_sum = np.array(adj.sum(1))
#    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
#   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#    return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def normalize_adj(adj):
    #adj = sp.coo_matrix(adj)
    #adj = adj + sp.eye(adj.shape[0]) #加自环
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt))  # D^-1/2AD^-1/2

def load_graph(data, args):
    Numnode = data.x.shape[0]
    structgraph_path = './CN_data/' + str(args.dataset) + +'/' + str(args.alpha) + '/c' + str(args.ks) + '.txt'
    struct_edges = np.genfromtxt(structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(Numnode, Numnode), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)#无自环
    #nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    sadj = sparse_mx_to_torch_sparse_tensor(normalize_adj(sadj))

    adj = edge_index_to_adj(data.edge_index) #无自环
    adj = sp.coo_matrix(adj)
    adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))

    return adj, sadj
