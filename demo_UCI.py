# # ----------------------
#  using simple data
# # ----------------------
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale, scale
from PlainRGCN import RandomizedGCN_Model
from utils import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
from EvoRGCN import EvoRGCN

import numpy as np
import time


# npz = np.load('D:\\Python\\Datasets\\UCI-Dataset-52.npz', allow_pickle=True)
# dataset = npz['data'][()]
# dataset_keys = list(dataset.keys())
# dataset_keys.sort()

'''
===================================
 MAIN: Train and Test Model 
===================================
'''

N_TRAIN = 0.1
REG_PARM = 1e-4

X, y = load_iris(return_X_y=True)
# X, y = dataset[dataset_keys]['x'], dataset[dataset_keys]['y']
classes = np.unique(y)
for c in classes:
    if np.nonzero(y == c)[0].shape[0] < 5:
        X = np.delete(X, np.nonzero(y == c), axis=0)
        y = np.delete(y, np.nonzero(y == c))
y = standardize_label(y)
X = minmax_scale(X)
print('X shape:', X.shape, 'n-class:', np.unique(y).shape[0])

acc_list = []
time_alg_i = []
for i in range(10):
    # # # Create Sparse Graph-structured dataset
    K_NEIGHBOR = 20
    # print(K_NEIGHBOR)
    sparse_g = kneighbors_graph(X, K_NEIGHBOR, include_self=False, mode='connectivity')
    g = sparse_g.todense()
    coo_g = sparse.coo_matrix(g)
    edge_index, edge_fea = from_scipy_sparse_matrix(coo_g)
    train_idx, test_idx = stratified_train_test_index(y, train_size=N_TRAIN, seed=i)
    train_mask = torch.zeros(X.shape[0])
    train_mask[train_idx] = 1
    train_mask = train_mask.type(torch.BoolTensor)
    test_mask = ~train_mask
    Y = encode_onehot(torch.from_numpy(y).type(torch.LongTensor))
    data = Data(x=torch.from_numpy(X).type(torch.FloatTensor), edge_index=edge_index,
                y=torch.from_numpy(y).to(torch.long),
                train_mask=train_mask, test_mask=test_mask, y_one_hot=Y)
    data.edge_index = to_undirected(data.edge_index)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    data.adj_t = gcn_norm(adj)
    data = T.NormalizeFeatures()(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # # =====================================
    #   Plain RGCN
    # # =====================================
    time_start = time.clock()
    model = RandomizedGCN_Model(regularization_coef=1e-5, residual=True, in_channel=X.shape[1], out_channel=5, device=device)
    y_pred = model(data)

    # # =====================================
    #   Evo RGCN
    # # =====================================
    # model = EvoRGCN(5, reg_para=1e-4, reg_para_1=1e-5, residual=False, neighbor_novelty=15, popsize=50, maxiter=50, n_novelty=5, device=device)
    # time_start = time.clock()
    # y_pred = model.fit(data)

    run_time = round(time.clock() - time_start, 4)
    acc = cal_test_acc(data.y_one_hot[data.test_mask], y_pred[data.test_mask])
    acc_list.append(acc.item())
    time_alg_i.append(run_time)
    print('run # %.4f, acc: %.4f, time: %.4f' % (i, acc, run_time))

print('\n\n===================== Summary =====================')
print('ALL Results:', acc_list)
print('Average: %.2f +- %.2f' % (np.mean(acc_list)*100, np.std(acc_list)*100))
print('time: %.4f' % np.asarray(time_alg_i).mean())

