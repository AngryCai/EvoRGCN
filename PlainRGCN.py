import torch
import torch_sparse as ts
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class RGC_Layer(MessagePassing):
    """
    random hidden layer with RVLF / ELM
    """

    def __init__(self, in_channels, out_channels, residual=False, cached=False, bias=False, hidden_w=None, **kwargs):
        super(RGC_Layer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.residual = residual
        self.hidden_w = hidden_w
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.uniform_(self.lin.weight, -1, 1)
        if self.hidden_w is None:
            torch.nn.init.normal_(self.lin.weight)
        elif self.hidden_w is not None and isinstance(self.hidden_w, torch.Tensor):
            self.hidden_w = self.hidden_w.float()
            self.lin.weight = torch.nn.Parameter(self.hidden_w)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        """"""
        x_ = x.clone().detach()
        x = self.lin(x)
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), False, True, dtype=x.dtype)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, norm=edge_weight)
        h = torch.sigmoid(x)
        if self.residual:
            h = torch.cat((h, x_), 1)
        x = self.propagate(edge_index, x=h, edge_weight=edge_weight, size=None, norm=edge_weight)
        if return_hidden:
            return x, h
        else:
            return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class RandomizedGCN_Model(torch.nn.Module):
    """
    train a plain RGCN with closed-form solution
    """

    def __init__(self, regularization_coef=1e-5,  residual=False, in_channel=None, out_channel=None, hidden_w=None, device='cpu', **kwargs):
        """
        :param regularization_coef:  non-negative regularization coefficient
        :param residual: bool, skip connection (direct links), GCRVFL: residual=True, GCELM: residual=Fasle
        :param in_channel: input feature dim
        :param out_channel: hidden layer dim, number of filters
        :param hidden_w: pre-define a hidden weights matrix or set to None for randomly generating
        :param device: 'cup' or 'cuda'
        :param kwargs:
        """
        super(RandomizedGCN_Model, self).__init__()
        self.regularization_coef = regularization_coef
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.residual = residual
        self.sgconv = RGC_Layer(in_channel, out_channel, residual=residual, cached=False, hidden_w=hidden_w).to(device)
        self.kwargs = kwargs

    def forward(self, data):
        """
        :param data: feed into a PyG Data that additionally holds a one-hot label matrix and a sparse adjacent matrix
        :return:
        """
        x_gconv = self.sgconv(data.x, data.edge_index, data.edge_attr)
        closed_form_solution = self.cal_closed_form_solution(x_gconv, data.y_one_hot, data.train_mask)
        y_pred = torch.matmul(x_gconv, closed_form_solution)
        self.closed_form_solution = closed_form_solution
        return y_pred

    def get_hidden(self, data):
        _, h = self.sgconv(data.x, data.edge_index, data.edge_attr, return_hidden=True)
        return h

    def cal_closed_form_solution(self, x_gconv, y_one_hot, train_mask):
        """
        :param x_gconv: dense tensor
        :param y_one_hot: dense label matrix
        :param train_mask: 1D dense tensor
        :return:
        """
        # mask = torch.diag(data.train_mask.float())
        # mask = ts.SparseTensor.eye(train_mask.size(0)).float()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        row = torch.arange(0, train_mask.shape[0], dtype=torch.long).to(self.device)
        col = torch.arange(0, train_mask.shape[0], dtype=torch.long).to(self.device)
        # val = train_mask.float().dense()
        mask = ts.SparseTensor(row=row, col=col, value=train_mask.float()).coalesce()
        Y_train = y_one_hot * torch.unsqueeze(train_mask, 1)
        # # support only (Sparse_A X Dense_B) or (Sparse_A X Sparse_B).
        # # Thus, for the case of (Dense_A X Sparse_B), considering (Sparse_B.T X Dense_A.T).T
        temp_a = ts.matmul(mask, x_gconv).transpose(1, 0)  # # X.T*M
        I = torch.eye(x_gconv.shape[1]).float().to(self.device)
        before_inv = torch.matmul(temp_a, x_gconv) + self.regularization_coef * I
        temp_left = torch.inverse(before_inv)
        temp_right = torch.matmul(temp_a, Y_train)
        solution = torch.matmul(temp_left, temp_right)
        return solution


