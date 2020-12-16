import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp
from torch.utils.data import Dataset
import pickle as pkl
#import random
from components import to_category, HfromDict, H2G
#from collections import OrderedDict as odict




def dt_read_all(datasetroot):
    train_batch_indexes = []
    test_batch_indexes = []
    with open(datasetroot + 'features.pickle', 'rb') as f1:
        fmat_all = pkl.load(f1).todense()#(number of nodes, dimension of features)=(2708, 1443)
    f1.close()
    with open(datasetroot + 'hypergraph.pickle', 'rb') as f2:
        Hdict = pkl.load(f2)# dict len = number of hyperedges = 1072
    f2.close()
    with open(datasetroot + 'labels.pickle', 'rb') as f3:
        labels = pkl.load(f3)#list len = number of nodes = 2708
    f3.close()
    # turn labels in form of csr_matrix into list
    #labels = _1hot(labels)
    # for i in range(10):
    #     path = datasetroot + '/splits/' + str(i + 1) + '.pickle'
    #     with open(path, 'rb') as f:
    #         ddict = pkl.load(f)
    #         train_batch_indexes.append(ddict['train'])
    #         test_batch_indexes.append(ddict['test'])

    return fmat_all, Hdict, labels
        #, train_batch_indexes, test_batch_indexes

def dtinit(datasetroot, with_test=False,  test_ratio=0.3):
    fmat_all, Hdict, labels = dt_read_all(datasetroot)
    H_all, name_index = HfromDict(Hdict)
    # split train and test sets f

    adj, Gfeat, Glabel = H2G(Hdict, fmat_all, labels)
    num_nodes_all = H_all.shape[0]
    num_hyper_edges_all = H_all.shape[1]
    feat_dim = fmat_all.shape[1]  # feat_dim is the same with Gfeat dimision
    # labels of original hypergraph
    if with_test:
        cut_idx = int(num_hyper_edges_all * (1 - test_ratio))
        adj_train = adj[:cut_idx, :cut_idx]
        Gfeat_train = Gfeat[:cut_idx, :]
        Glabel_train = Glabel[:cut_idx, :]
        return adj_train, Gfeat_train, Glabel_train, adj, Gfeat, Glabel
    else:
        return adj, Gfeat, Glabel

class HGraphSequenceRandSet(Dataset):
    def __init__(self, adj, X, label, num_permutation=400, seed=None, fix=False):
        super(HGraphSequenceRandSet, self).__init__()
        """
        mode: a string to be chosen between 'iso' and 'exist'
        """
        self.seed = seed
        self.adj = adj
        self.len = adj.shape[0]
        self.X = X
        self.label = label
        self.num_permutation = num_permutation
        self.fix = fix


    # def sort_split(self):
    #     Hlistnew = sorted(self.Hdict, key= lambda k:self.Hdict[k][0])
    #     train_list = Hlistnew[:len(Hlistnew) - self.num_test]
    #     test_list = Hlistnew[self.num_test:]
    #     return train_list, test_list
    #
    # def gen_split(self):
    #     keylist = self.Hdict.keys()
    #     train_list = keylist[:len(keylist) - self.num_test]
    #     test_list = keylist[self.num_test:]
    #     return train_list, test_list

    def __len__(self):
        return self.num_permutation

    def __getitem__(self, idx):
        adj_copy = self.adj
        feat_copy = self.X
        label_copy = to_category(self.label)

        if not self.fix:
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            feat_copy = feat_copy[x_idx, :]
            label_copy = label_copy[x_idx]

        #adj_copy = adj_copy.astype(np.float32)
        #feat_copy = feat_copy.astype(np.float32)
        #label_copy = label_copy.astype(np.float32)

        #return torch.from_numpy(adj_copy), torch.from_numpy(feat_copy), torch.from_numpy(label_copy)
        return adj_copy, feat_copy, label_copy

if __name__ == "__main__":
    datasetroot = '../data/coauthorship/cora/'
    #Hset = HGraphSequenceTrainSet(datasetroot)
    fmat_all, Hdict, labels = dt_read_all(datasetroot)

    print('a')






