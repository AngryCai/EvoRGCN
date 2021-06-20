from EvoRGCN import EvoRGCN
from PlainRGCN import RandomizedGCN_Model
from utils import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import torch
import time

# # Cora, PubMed, CiteSeer
data_name = 'Cora'  # # settings: SGC: regularization_coef=2, K_hop=2; APPNP: alpha=0.1, K=20
# data_name = 'CiteSeer'  # # settings: SGC: regularization_coef=30, K_hop=2; APPNP: alpha=0.1, K=20
# data_name = 'PubMed'  # # settings: regularization_coef=0.02, K_hop=2; APPNP: alpha=0.05, K=15

root = '/tmp/' + data_name
score_acc = []
score_f1 = []
time_list = []
for i in range(10):
    data = load_data(root, data_name, split='public')
    # data.train_mask = ~(data.test_mask + data.val_mask)

    print('num-train: %s, num-val: %s, num-test: %s' %
          (data.train_mask.sum().item(), data.val_mask.sum().item(), data.test_mask.sum().item()))

    # # reduce dim using PCA
    pca = PCA(n_components=100)
    x_pca = pca.fit_transform(data.x.numpy())
    data.x = torch.from_numpy(x_pca)
    print('pca shape: %s, percentage: %s' % (x_pca.shape, np.sum(pca.explained_variance_ratio_)))

    in_channel = data.x.shape[1]
    n_hidden = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # # ==================================
    # # NON-Evolutionary Baselines
    model = RandomizedGCN_Model(regularization_coef=2, residual=True, in_channel=in_channel, out_channel=n_hidden, device=device)
    # model = SGC_Model(lr=0.01, epoch=100, weight_decay=0.01, K_hop=2, verbose=False)
    # model = GCN_Model([50], lr=0.01, epoch=100, weight_decay=0.0004, is_linear=False, verbose=False)
    start_time = time.time()
    y_pred = model(data)
    acc_test = cal_test_acc(data.y_one_hot[data.test_mask], y_pred[data.test_mask]).item()

    # # ==================================
    # # Evolutionary GCELM/GCRVFL
    # model = EvoRGCN(n_hidden, reg_para=10, reg_para_1=1e-4, residual=True, neighbor_novelty=15, popsize=50, maxiter=50, n_novelty=5, device=device)
    # start_time = time.time()
    # y_pred = model.fit(data)
    # acc_test = cal_test_acc(data.y_one_hot[data.test_mask], y_pred[data.test_mask]).item()

    # # ==================================
    # # ELM/RVFL on non-structural data
    # data = data.to('cpu')
    # x = data.x.numpy()
    # y = data.y_one_hot.numpy()
    # x_train, y_train = x[data.train_mask], y[data.train_mask]
    # x_test, y_test = x[data.test_mask], y[data.test_mask]
    # start_time = time.time()
    # # model = BaseELM(n_hidden, reg=50)
    # model = RVFL(n_hidden, reg=50)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # acc_test = accuracy_score(y_test.argmax(axis=1), y_pred)

    run_time = time.time() - start_time
    print('ACC: %.4f' % acc_test)
    score_acc.append(acc_test)
    time_list.append(run_time)

print('===========================================')
print('ACC: {:.2f} +- {:.2f}'.format(np.mean(score_acc)*100, np.std(score_acc)*100))
print('TIME: {:.4f}s'.format(np.mean(time_list)))


