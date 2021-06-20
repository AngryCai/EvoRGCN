import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import minmax_scale
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, GDC
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from Toolbox.Preprocessing import Processor


def encode_onehot(labels):
    n_clz = torch.unique(labels).size(0)
    source = torch.ones((labels.shape[0], 1), dtype=torch.float32)
    labels_onehot = torch.zeros((labels.shape[0], n_clz), dtype=torch.float32)
    labels_onehot.scatter_(dim=1, index=labels.unsqueeze(1), src=source)
    return labels_onehot


def load_data(path="../data/cora/", dataset_name="cora", use_dgc=False, split='public'):
    """
    Load dataset from PyG datasets
    :param path:
    :param dataset_name:
    :param use_dgc: whether DGC is being used
    :return:
    """
    # path='/tmp/Cora'
    # dataset_name='Cora'
    print('Loading {} dataset...'.format(dataset_name))
    dataset = Planetoid(root=path, name=dataset_name, transform=[NormalizeFeatures()], split=split)
    data = dataset.data
    if use_dgc:
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.05),
                  sparsification_kwargs=dict(method='topk', k=128,
                                             dim=0), exact=True)
        data = gdc(data)
    print(data)
    adj = sp.coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                        shape=(data.num_nodes, data.num_nodes),
                        dtype=np.float32)
    y_one_hot = encode_onehot(data.y)
    data.y_one_hot = y_one_hot
    data.adj = adj
    return data


def cal_test_acc(y_test, y_pred):
    acc = torch.eq(torch.argmax(y_test, dim=1), torch.argmax(y_pred, dim=1))
    acc = torch.sum(acc) * 1. / y_test.shape[0]
    return acc


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN_Model model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def prepare_graph_from_regular_data(X, y, neighbor, train_size=0.1, seed=None):
    X = minmax_scale(X)
    p = Processor()
    y = p.standardize_label(y)
    sparse_g = kneighbors_graph(X, neighbor, include_self=False, mode='connectivity')
    g = sparse_g.todense()
    coo_g = sparse.coo_matrix(g)
    edge_index, edge_fea = from_scipy_sparse_matrix(coo_g)
    train_idx, test_idx = p.stratified_train_test_index(y, train_size=train_size, seed=seed)
    train_mask = torch.zeros(X.shape[0])
    train_mask[train_idx] = 1
    train_mask = train_mask.type(torch.BoolTensor)
    test_mask = ~train_mask
    Y = encode_onehot(torch.from_numpy(y).type(torch.LongTensor))
    data = Data(x=torch.from_numpy(X).type(torch.FloatTensor), edge_index=edge_index,
                y=torch.from_numpy(y).to(torch.long),
                train_mask=train_mask, test_mask=test_mask, y_one_hot=Y)
    data.edge_index = to_undirected(data.edge_index)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    data.adj_t = gcn_norm(adj)
    return data


def standardize_label(y):
    """
    standardize the classes label into 0-k
    :param y:
    :return:
    """
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


def stratified_train_test_index(y, train_size, seed=None):
    """
    :param y: labels
    :param train_size: int, absolute number for each classes; float [0., 1.], percentage of each classes
    :return:
    """
    np.random.seed(seed)
    train_idx, test_idx = [], []
    for i in np.unique(y):
        idx = np.nonzero(y == i)[0]
        np.random.shuffle(idx)
        num = np.sum(y == i)
        if 0. < train_size < 1.:
            train_size_ = int(np.ceil(train_size * num))
        elif train_size > num or train_size <= 0.:
            raise Exception('Invalid training size.')
        else:
            train_size_ = np.copy(train_size)
        train_idx += idx[:train_size_].tolist()
        test_idx += idx[train_size_:].tolist()
    train_idx = np.asarray(train_idx).reshape(-1)
    test_idx = np.asarray(test_idx).reshape(-1)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx
