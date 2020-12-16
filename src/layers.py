import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

from torch.autograd import Variable
from components import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False, act=lambda x: x, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, training = self.training)
        input = torch.squeeze(input)
        ishape = input.shape
        ashape = adj.shape
        wshape = self.weight.shape
        support = torch.mm(input, self.weight)
        #try:
        output = torch.spmm(adj, support)
        #except:
         #   p = 1
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_classifier(Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, dropout=0.2):
        super(GCN_classifier, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(self.feature_dim, self.hidden_dim, dropout=self.dropout, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.out_dim)


    def forward(self, adj, X):
        hidden = self.gc1(X, adj)
        out = self.gc2(hidden, adj)
        return F.log_softmax(out, dim=1) # corresponding to NLL loss
    
class GCN_regress(Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, mlp=False,  dropout=0.2):
        super(GCN_regress, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(self.feature_dim, self.hidden_dim, dropout=self.dropout, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.out_dim)
        if mlp:
            self.linear = MLP(out_dim, out_dim/2, 1, bias=True, dropout=0.0)
        else:
            self.linear = nn.Linear(out_dim, 1)

    def forward(self, adj, X):
        hidden = self.gc1(X, adj)
        out = self.gc2(X, adj)
        return self.linear(out)

class wTransitionLinearUnit(Module):
    def __init__(self, ori_dim, tar_dim):
        super(wTransitionLinearUnit, self).__init__()
        self.linear_1 = torch.nn.Linear(tar_dim, ori_dim)
        self.linear_2 = torch.nn.Linear(tar_dim, tar_dim)
        self.linear_3 = torch.nn.Linear(tar_dim, tar_dim)
        self.ori_dim = ori_dim
        self.tar_dim = tar_dim

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # torch.nn.init.xavier_normal_(self.weight)
        # self.bias.data.uniform_(-stdv, stdv)
        pass

    def forward(self, last_w, z_cov):
        z_cov = (z_cov - z_cov.mean())/ z_cov.std()
        hidden = F.relu(self.linear_1(z_cov.t()))
        w_update = self.linear_2(hidden)
        update_gate = torch.sigmoid(self.linear_3(hidden))

        w_update = torch.clamp(w_update, min=-0.1, max=0.1)#why

        return (1 - update_gate) * last_w + w_update * update_gate

class recurHGC_add(Module):
    """
    Given H, X, and X_new(with new hyper-edge), sample Z_new from P(Z_new| H_hat, X_new)
    """
    def __init__(self, features_dim, hidden_dim, out_dim, random_add=False, dropout=0.3):
        super(recurHGC_add, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.random_add = random_add

        # self.hgc1 = HGconv(features_dim, hidden_dim, act=F.relu)
        # self.hgc_mean = HGconv(hidden_dim, out_dim)
        # self.hgc_log_std = HGconv(hidden_dim, out_dim)
        self.gc1 = GraphConvolution(features_dim, hidden_dim, dropout=dropout, act=F.relu)
        self.gc_mean = GraphConvolution(hidden_dim, out_dim,  dropout=dropout)
        self.gc_log_std = GraphConvolution(hidden_dim, out_dim,  dropout=dropout)

    def forward(self, adj, input, input_new=None):
        # input is the corresponding features matrix to the incidence matrix
        # if input_new is None:
        #     H = H.numpy()
        #     hidden = self.hgc1(input, H)
        #     z_mean = self.hgc_mean(hidden, H)
        #     z_log_std = self.hgc_log_std(hidden, H)
        #     return z_mean, z_log_std
        # num_total_nodes = input_new.size()[0] + input.size()[0]
        # # num_total_nodes =
        # if self.training:
        #     with torch.no_grad():
        #         H_new = torch.zeros(num_total_edges, num_total_nodes)
        if input_new is None:
            adj = adj.numpy()
            adj_norm = anormalize(adj)
            hidden = self.gc1(input, adj_norm)
            z_mean = self.gc_mean(hidden, adj_norm)
            z_log_std = self.gc_log_std(hidden, adj_norm)
            return z_mean, z_log_std

        num_total_nodes = input_new.size()[0] + input.size()[0]
        if self.training:
            with torch.no_grad():
                adj_new = torch.zeros(num_total_nodes, num_total_nodes)
                if self.random_add:
                    num_edges = float(((adj > 0).sum() - adj.shape[0]) / 2)
                    p0 = num_edges / (num_total_nodes ** 2)
                    adj_new.bernoulli_(p0)
                    adj_new = adj_new - adj_new.tril()
                    adj_new = ((adj_new + adj_new.t()) > 0).float()

                adj_new += torch.eye(num_total_nodes)
                adj_new[:input.size()[0], :input.size()[0]] = adj
                adj_new = adj_new.numpy()
                adj_norm = anormalize(adj_new)
                input = torch.squeeze(input)
                input_new = torch.squeeze(input_new)
                input_all = torch.cat((input, input_new))

            hidden = self.gc1(input_all, adj_norm)
            z_mean = self.gc_mean(hidden, adj_norm)
            z_log_std = self.gc_log_std(hidden, adj_norm)

            z_mean_old = z_mean[:input.size()[0], :]
            z_log_std_old = z_log_std[:input.size()[0], :]

            try:
                assert input.size()[0] < input_all.size()[0]
            except:
                p = 1
            z_mean_new = z_mean[input.size()[0]:, :]
            # try:
            #     assert z_mean_new != []
            # except:
            #     p = 1

            z_log_std_new = z_log_std[input.size()[0]:, :]

            return z_mean_old, z_log_std_old, z_mean_new, z_log_std_new

        else:
            adj = adj.numpy()
            adj_norm = anormalize(adj)
            hidden_old = self.gc1(input, adj_norm)
            z_mean_old = self.gc_mean(hidden_old, adj_norm)
            z_log_std_old = self.gc_log_std(hidden_old, adj_norm)

            adj_new = torch.eye(input_new.size()[0])
            hidden_new = self.gc1(input_new, adj_new)
            z_mean_new = self.gc_mean(hidden_new, adj_new)
            z_log_std_new = self.gc_log_std(hidden_new, adj_new)
            return z_mean_old, z_log_std_old, z_mean_new, z_log_std_new

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphVae(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=False, dropout=0.3):
        super(GraphVae, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.gc1 = GraphConvolution(features_dim, hidden_dim, bias=bias, dropout=dropout, act=F.relu)
        self.gc_mean = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)
        self.gc_log_std = GraphConvolution(hidden_dim, out_dim, bias=bias, dropout=dropout)

    def forward(self, adj, input):
        hidden = self.gc1(input, adj)
        z_mean = self.gc_mean(hidden, adj)
        z_log_std = self.gc_log_std(hidden, adj)
        return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(Module):
    def __init__(self, features_dim, hidden_dim, out_dim, bias=True, dropout=0.3):
        super(MLP, self).__init__()
        self.features_dim = features_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.linear = torch.nn.Linear(features_dim, hidden_dim)
        self.z_mean = torch.nn.Linear(hidden_dim, out_dim)
        self.z_log_std = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        hidden = F.relu(self.linear(input))
        z_mean = F.dropout(self.z_mean(hidden), self.dropout, training=self.training)
        z_log_std = F.dropout(self.z_log_std(hidden), self.dropout, training=self.training)
        return z_mean, z_log_std

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# no
# class recurHGCN(Module):
#     def __init__(self, in_features, hidden_dim, out_features, bias=True, dropout=0.3):
#         """
#
#         :param in_features:
#         :param hidden_dim:
#         :param out_features:
#         :param bias:
#         :param dropout:
#         """
#         super(recurHGCN, self).__init__()
#
#         self.dropout = dropout
#
#         self.init_hidden_weight = Parameter(torch.FloatTensor(in_features, hidden_dim))
#         self.init_hidden_bias = Parameter(torch.FloatTensor(hidden_dim))
#
#         self.init_mean_weight = Parameter(torch.FloatTensor(hidden_dim, out_features))
#         self.init_mean_bias = Parameter(torch.FloatTensor(out_features))
#
#         self.init_log_std_weight = Parameter(torch.FloatTensor(hidden_dim, out_features))
#         self.init_log_std_bias = Parameter(torch.FloatTensor(out_features))
#
#         self.hidden_w_transition = wTransitionLinearUnit(in_features, hidden_dim)
#         self.mean_w_transition = wTransitionLinearUnit(hidden_dim, out_features)
#         self.log_std_w_transition = wTransitionLinearUnit(hidden_dim, out_features)
#
#         self.reset_parameters()
#
#     def init_all_weights(self):# why is there such
#         self.hidden_weight = self.init_hidden_weight + 0.0
#         self.hidden_bias = self.init_hidden_bias + 0.0
#
#         self.mean_weight = self.init_mean_weight + 0.0
#         self.mean_bias = self.init_mean_bias + 0.0
#
#         self.log_std_weight = self.init_log_std_weight + 0.0
#         self.log_std_bias = self.init_log_std_bias + 0.0
#
#     def conv_ops(self, input, adj):
#         input = F.dropout(input, self.dropout, training=self.training)
#         support = torch.mm(input, self.hidden_weight)
#         hidden = F.relu(torch.spmm(adj, support) + self.hidden_bias)
#
#         hidden = F.dropout(hidden, self.dropout, training=self.training)
#         support_mean = torch.mm(hidden, self.mean_weight)
#         mean = torch.spmm(adj, support_mean)
#         support_std = torch.mm(hidden, self.log_std_weight)
#         log_std = torch.spmm(adj, support_std)
#
#         return mean, log_std
#
#     def weight_transition(self, last_z, current_z):
#         # compute the 'covariance' matrix for the difference of z
#         z_diff = last_z - current_z
#         z_cov = torch.mm(torch.t(z_diff), z_diff)
#         # self.hidden_weight = self.hidden_w_transition(self.hidden_weight, z_cov) * 1.0
#         self.mean_weight = self.mean_w_transition(self.mean_weight, z_cov) * 1.0
#         self.log_std_weight = self.log_std_w_transition(self.log_std_weight, z_cov) * 1.0
#
#     def forward(self, adj, input, update_size, input_new=None):
#         print(adj.size(0))
#         print(update_size)
#         if adj.size(0) < update_size:
#             raise ValueError('adj must be no less than update size!')
#
#         self.init_all_weights()
#
#         normal = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(1.0))

# no
# class hyperGraphVAE(Module):
#     def __init__(self, features_dim, hidden_dim, out_dim, bias=False, dropout=0.3):
#         super(hyperGraphVAE, self).__init__()
#         self.features_dim = features_dim
#         self.out_dim = out_dim
#         self.dropout = dropout
#
#         self.hgc1 = HGconv(features_dim, hidden_dim, act=F.relu)
#         self.hgc_mean = HGconv(hidden_dim, out_dim)
#         self.hgc_log_std = HGconv(hidden_dim, out_dim)
#
#     def forward(self, input, H):
#         hidden = self.hgc1(input, H)
#         z_mean = self.hgc_mean(hidden, H)
#         z_log_std = self.hgc_log_std(H, hidden)
#
#         return z_mean, z_log_std
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


# class hyperGraphAE(Module):
#     def __init__(self, features_dim, hidden_dim, out_dim):
#         super(hyperGraphAE, self).__init__()
#         self.features_dim = features_dim
#         self.out_dim = out_dim
#
#         self.hgc1 = HGconv(features_dim, hidden_dim, act=F.relu)
#         self.hgc_z = HGconv(hidden_dim, out_dim)
#
#     def forward(self, input, H):
#         hidden = self.hgc1(input, H)
#         z = self.hgc_z(hidden, H)
#         return z
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features)

# no
# class HGCN(nn.Module):
#     def __init__(self, V, E, X, args):
#         """
#         :param V: vertex set
#         :param E:
#         :param X:
#         :param args:
#             d : initial node feature dimension
#             h : number of hidden units
#             c : number of classes
#         """
#         super(HGCN, self).__init__()
#         d, l, c = args.d, args.depth, args.c
#         cuda = args.cuda and torch.cuda.is_available()
#
#         h = [d]
#         for i in range(l - 1):
#             power = l - i + 2
#             if args.dataset == 'citeseer': power = l - i + 4
#             h.append(2 ** power)
#         h.append(c)
#
#         if args.fast:
#             reapproximate = False
#             structure = _components.Laplacian(V, E, X, args.mediators)
#         else:
#             reapproximate = True
#             structure = E
#
#         self.layers = nn.ModuleList(
#             [_components.HGconv(h[i], h[i + 1], reapproximate, cuda) for i in range(l)])
#         self.do, self.l = args.dropout, args.depth
#         self.structure, self.m = structure, args.mediators
#
#     def forward(self, H):
#         do, l, m = self.do, self.l, self.m
#         for i, hidden in enumerate(self.layers):
#             H = F.relu(hidden(self.structure, H, m))
#             if i < l - 1:
#                 V = H
#                 H = F.dropout(H, do, training=self.training)
#
#         return F.log_softmax(H, dim=1)