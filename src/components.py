import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from _components import *
from scipy.sparse import *


def _1hot(labels):
    """
    converts each positive integer (representing a unique class) into ints one-hot form
    csr_mat --> 1hot numpyarray

    Arguments:
    labels: a list of positive integers with eah integer representing a unique label
    """

    classes = set(labels)
    onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(onehot.get, labels)), dtype=np.int32)

def to_category(labels):
    """
    converts 1hot array to categorical
    """
    clabels = torch.nonzero(labels)
    clabels = torch.squeeze(clabels)
    clabels = clabels[:,1]
    clabels = torch.squeeze(clabels)
    return clabels


def HfromDict(hdict):
    # not efficient
    node_list_rep = []
    name_index = {}
    name_index_inv = {}
    i = 0
    for key, item in hdict.items():
        node_list_rep += item
        name_index.update({key: i})
        #name_index_inv.update({i, key})
        #name_index.append(key)
        i += 1
    #node_list = list(set(node_list_rep))
    max_nodes = max(node_list_rep)
    H = torch.zeros(max_nodes+1, len(hdict))
    for k, l in hdict.items():
        for node_id in l:
            H[node_id, name_index[k]] = 1
    return H, name_index

    # row = []
    # col = []
    # i = 0
    # for k in Hdict.keys:
    #     name_index[k] = i
    #     row_num_node = len(Hdict[k])
    #     current_row = [i for j in range(len(row_num_node))]
    #     row.append(current_row)
    #     i += 1
    # col.append(Hdict[k])
    # assert len(col) == len(row)
    # data = [1 for j in range(len(col))]
    # H_sparse = csr_matrix((data, (row, col)), shape=(3,3))


def H2G(Hdict, fmat_all, glabels):
    """
    :param H tensor indicating the incidence matrix of a hypergraph(num of nodes, num of hyperedges)
           fmat_all: feature matrix of all the nodes
    :return: adj, X(feature matrix of size(#nodes, dim))
    """
    H, name_index = HfromDict(Hdict)
    fmat_all = torch.Tensor(fmat_all)
    adj = torch.zeros(H.shape[1], H.shape[1])
    glabels = _1hot(glabels)
    glabels = torch.Tensor(glabels)
    labels = np.zeros((int(H.shape[1]), int(glabels.shape[1])))
    X = torch.zeros(H.shape[1], fmat_all.shape[1])
    i = 0
    for key, list in Hdict.items():
        # For adj
        for j in range(i+1, H.shape[1]):
            #add = int(torch.sum(H, dim=1)[i]) + int(torch.sum(H, dim=1)[j])
            ids1 = set(H[:,i].nonzero())
            ids2 = set(H[:,j].nonzero())
            _union = ids1.union(ids2)
            union = len(_union)
            _inter = ids1.intersection(ids2)
            inter = len(_inter)
            adj[i, j] = inter / union
        adj[i, i] += 1
        # For features
        numv = sum(H[:, i])
        nz_ids = H[:, i].nonzero()
        veall = torch.sum(fmat_all[nz_ids, :], axis=0)
        veall = torch.Tensor(veall)
        X[i, :] = torch.sum(veall, 0)/ numv
        # For labels
        labels_i = torch.zeros(len(list), glabels.shape[1])
        for j in range(len(list)):
            labels_i[j, :] = glabels[list[j], :]
        label_sum = torch.sum(labels_i, dim=0)
        id_max = torch.argmax(label_sum)
        id_max = id_max.numpy()
        #tid = torch.tensor(i, dtype=torch.uint8)
        #id_max = id_max.type(torch.uint8)
        #fillind = (torch.LongTensor(tid), torch.LongTensor(id_max))
        #labels.index_put_(fillind, torch.tensor([1.]))
        labels[i, id_max] = 1
        i += 1
    labels = torch.Tensor(labels)
    return adj, X, labels

def anormalize(adj):
    if type(adj) == torch.Tensor:
        adj = adj.numpy()
    ashape = adj.size
    try:
        rowsum = np.sum(adj, axis=1)
    except:
        p = 1
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    #degree_mat_inv_sqrt = torch.Tensor(degree_mat_inv_sqrt)
    #rowsum = torch.Tensor(rowsum)
    #rshape = rowsum.shape
    #ashape = adj.shape
    #dshape = degree_mat_inv_sqrt.shape
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.Tensor(adj_normalized.astype(np.float32))


# if __name__ == "__main__":
#     import pickle as pkl
#     datasetroot = '../data/coauthorship/cora/'
#     with open(datasetroot + 'features.pickle', 'rb') as f1:
#         fmat_all = pkl.load(f1).todense()  # (number of nodes, dimension of features)=(2708, 1443)
#     f1.close()
#     with open(datasetroot + 'hypergraph.pickle', 'rb') as f2:
#         Hdict = pkl.load(f2)  # dict len = number of hyperedges = 1072
#     f2.close()
#     with open(datasetroot + 'labels.pickle', 'rb') as f3:
#         glabels = pkl.load(f3)  # list len = number of nodes = 2708
#     f3.close()
#     H = HfromDict(Hdict)
#     adj, X, labels = H2G(Hdict, fmat_all, glabels)
#     print(adj.shape)




