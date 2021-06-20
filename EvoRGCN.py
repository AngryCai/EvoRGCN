import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

from JADE import minimize
from PlainRGCN import RandomizedGCN_Model


class EvoRGCN:
    """
    optimize plain RGCN by using JADE-NS or any other predefined heuristic optimizers.
    """
    def __init__(self, n_hidden, reg_para=1., reg_para_1=1., residual=False, neighbor_novelty=15, individual_boundary=(-1, 1), popsize=50, maxiter=100, n_novelty=5, device='cpu'):
        """
        :param n_hidden: number of hidden neurons/filters/channels
        :param reg_para: regularization coefficient
        :param reg_para_1: coefficient in the fitness function
        :param residual: True: GCRVFL, False: GCELM
        :param neighbor_novelty: NS hyperparameter, default 15
        :param individual_boundary: boundary of solutions
        :param popsize: population size, default 50
        :param maxiter: maximum generation
        :param n_novelty: number of NS individuals
        :param device: 'cup' or 'cuda'
        """
        self.neighbor_novelty = neighbor_novelty
        self.reg_para_1 = reg_para_1
        self.reg_para = reg_para
        self.n_hidden = n_hidden
        self.residual = residual
        self.popsize = popsize
        self.maxiter = maxiter
        self.device = device
        self.individual_boundary = individual_boundary
        self.n_novelty = n_novelty

    def fit(self, data):
        self.data = data.to(self.device)
        n_paras = data.x.shape[1] * self.n_hidden
        self.in_channel = data.x.shape[1]
        bounds = [self.individual_boundary]*n_paras   # bounds [(x1_min, x1_max), (x2_min, x2_max),...]
        best_W = minimize(self.fitness_func, bounds, self.popsize, self.maxiter,
                          n_novelty_solution=self.n_novelty, n_neighbors=self.neighbor_novelty)
        best_W_hidden = torch.from_numpy(best_W.reshape((self.data.x.shape[1], self.n_hidden)).transpose())

        best_model = RandomizedGCN_Model(regularization_coef=self.reg_para, residual=self.residual,
                                         in_channel=self.in_channel, out_channel=self.n_hidden,
                                         hidden_w=best_W_hidden, device=self.device)
        self.best_model = best_model
        self.y_pred = best_model(self.data)
        return self.y_pred

    def fitness_func(self, W):
        W_hidden = np.asarray(W)
        W_hidden = W_hidden.reshape((self.data.x.shape[1], self.n_hidden)).transpose()
        W_hidden = torch.from_numpy(W_hidden)
        model = RandomizedGCN_Model(regularization_coef=self.reg_para, residual=self.residual,
                                    in_channel=self.in_channel, out_channel=self.n_hidden, hidden_w=W_hidden, device=self.device)
        y_pred = model(self.data)
        err = torch.nn.MSELoss()(self.data.y_one_hot[self.data.train_mask], y_pred[self.data.train_mask]).item()
        W_norm = np.linalg.norm(W_hidden.reshape(-1), ord=2)
        fitness = err + self.reg_para_1 * W_norm
        return fitness

    def predict(self, X=None):
        self.label = self.y_pred[self.data.test_mask].item().argmax(axis=1)
        return self.label

    def __adjacent_mat(self, x, n_neighbors=10):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A + A.T
        A[np.where(A == 2)] = 1
        A[np.diag_indices_from(A)] = 2
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A


# '''
# ===================================
#  MAIN: Train and Test Model
# ===================================
# '''
# import numpy as np
# import sklearn.datasets as dt
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import label_binarize, minmax_scale, normalize
# from Toolbox.Preprocessing import Processor
#
# N_TRAIN = 5
# p_Cora = Processor()
# X, y = dt.load_iris(return_X_y=True)
#
# X = normalize(X)
# train_idx, test_idx = p_Cora.stratified_train_test_index(y, train_size=N_TRAIN)
# y_train, y_test = y[train_idx], y[test_idx]
# X_train, X_test = X[train_idx], X[test_idx]
# Y = label_binarize(y, np.unique(y))
# Y_train, Y_test = Y[train_idx], Y[test_idx]
#
# gcelm = NSDE_GCELM(50, reg_para=1e-8, neighbor_gcelm=5, neighbor_novelty=5, popsize=50, maxiter=50, n_novelty=5)
# gcelm.fit(X_train, Y_train, X_test, Y_test)
# y_pre = gcelm.predict()
# acc = accuracy_score(y_test, y_pre)
# print('Our ACC=', acc)
