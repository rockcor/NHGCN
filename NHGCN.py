import torch
from torch.nn import Dropout, Parameter, Softmax, Sigmoid
from torch.nn.init import xavier_uniform_, constant_, xavier_uniform_, calculate_gain
from torch_geometric.nn import GCNConv,Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, matmul, mul, spspmm, remove_diag
from torch_sparse import sum as sparsesum

class NHGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()
        self.W1L = Parameter(torch.empty(num_features, params.hidden))
        self.W1H = Parameter(torch.empty(num_features, params.hidden))
        self.W2L = Parameter(torch.empty(params.hidden, params.hidden))
        self.W2H = Parameter(torch.empty(params.hidden, params.hidden))

        self.lam = Parameter(torch.zeros(3))
        self.lam1 = Parameter(torch.zeros(2))
        self.lam2 = Parameter(torch.zeros(2))
        self.dropout = Dropout(p=params.dropout)
        self.dropout2 = Dropout(p=params.dropout)
        self.finaldp = Dropout(p=params.finaldp)
        self.act = F.relu

        self.WX = Parameter(torch.empty(num_features, params.hidden))
        # self.lin2 = Linear(3 * params.hidden, num_classes,bias=False)
        self.lin1 = Linear(params.hidden, num_classes)
        self.args = params
        self._cached_adj_l = None
        self._cached_adj_h = None
        self.reset_parameter()

    def reset_parameter(self):
        xavier_uniform_(self.W1L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W1H, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2H, gain=calculate_gain('relu'))
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))

    def agg_norm(self, adj_t, mask, mtype='target'):
        # TODO: A^2
        if mtype == 'target':
            A_tilde = mul(adj_t,mask.view(-1,1))
        elif mtype == 'source':
            A_tilde = mul(adj_t,mask.view(1,-1))
        else:
            A_tilde = SparseTensor.from_torch_sparse_coo_tensor(
                torch.sparse.mm(
                    mask, torch.sparse.mm(
                        mask, adj_t.to_torch_sparse_coo_tensor())))
        if self.args.addself:
            A_tilde = fill_diag(A_tilde, 1.)
        else:
            A_tilde = remove_diag(A_tilde)
        D_tilde = sparsesum(A_tilde, dim=1)
        D_tilde_sq = D_tilde.pow_(-0.5)
        D_tilde_sq.masked_fill_(D_tilde_sq == float('inf'), 0.)
        A_hat = mul(A_tilde, D_tilde_sq.view(-1, 1))
        A_hat = mul(A_hat, D_tilde_sq.view(1, -1))

        return A_hat

    def forward(self, data):
        x = SparseTensor.from_dense(data.x)
        cc_mask = data.cc_mask
        # cc_mask_t = torch.unsqueeze(data.cc_mask, dim=-1)
        rev_cc_mask = torch.ones_like(cc_mask) - cc_mask
        # rev_cc_mask = 1 / (cc_mask + 1)
        # rev_cc_mask_t = torch.unsqueeze(rev_cc_mask, dim=-1)
        edge_index = data.edge_index
        adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])

        # low_cc mask
        if data.update_cc:
            A_hat_l = self.agg_norm(adj_t, cc_mask, 'source')
            self._cached_adj_l = A_hat_l
        else:
            A_hat_l = self._cached_adj_l

        # high_cc mask
        if data.update_cc:
            A_hat_h = self.agg_norm(adj_t, rev_cc_mask, 'source')
            self._cached_adj_h = A_hat_h
        else:
            A_hat_h = self._cached_adj_h

        xl = matmul(A_hat_l, x)
        xl = matmul(xl, self.W1L)
        xl = self.act(xl)
        xl = self.dropout(xl)
        xl = torch.mm(matmul(A_hat_l, xl), self.W2L)
        # high_cc partion
        xh = matmul(A_hat_h, x)
        xh = matmul(xh, self.W1H)
        xh = self.act(xh)
        xh = self.dropout(xh)
        xh = torch.mm(matmul(A_hat_h, xh), self.W2H)

        x = matmul(x, self.WX)
        x = self.act(xh)
        x = self.dropout(xh)

        lamx, laml, lamh = Softmax()(self.lam)
        if self.args.finalagg == 'add':
            xf = lamx * x + laml * xl + lamh * xh
            xf = self.act(xf)
            xf = self.finaldp(xf)
            xf = self.lin1(xf)
        elif self.args.finalagg == 'max':
            xf = torch.stack((x, xl, xh), dim=0)
            xf = torch.max(xf, dim=0)[0]
            xf = self.act(xf)
            xf = self.finaldp(xf)
            xf = self.lin1(xf)


        return xf
