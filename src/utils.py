import numpy as np
import torch
from sklearn import metrics
import pickle

def sample_reconstruction(z_mean, z_log_std):
    # using gumbel softmax
    num_nodes = z_mean.size()[0]
    normal = torch.distributions.Normal(0, 1)

    #sample z to approximate the posterior of A
    z = normal.sample(z_mean.size())
    z = z * torch.exp(z_log_std) + z_mean
    adj_h = torch.mm(z, z.permute(1, 0))
    return adj_h

def get_roc_auc_score(H, H_h, mask):
    H_n = H[mask].numpy() > 0.9
    H_h_n = H_h[mask].sigmoid().numpy()
    return metrics.roc_auc_score(H_n, H_h_n)

def get_average_precision_score(H, H_h, mask):
    H_n = H[mask].numpy() > 0.9
    H_h_n = H_h[mask].sigmoid().numpy()
    return metrics.average_precision_score(H_n, H_h_n)

def get_equal_mask(H_true, test_mask, thresh=0):
    H_true = H_true > thresh
    #TODO modify the test_mask
    pos_link_mask = H_true * test_mask
    num_links = int(pos_link_mask.sum().item())

    if num_links > 0.5 * test_mask.sum().item():
        raise  ValueError('test nodes, over connected!')

    neg_link_mask = ~H_true * test_mask
    neg_link_mask = neg_link_mask.numpy()
    row, col = np.where(neg_link_mask > 0)
    new_idx = np.random.permutation(len(row))
    row, col = row[new_idx][:num_links], col[new_idx][:num_links]
    neg_link_mask *= 0
    neg_link_mask[row, col] = 1
    neg_link_mask = torch.from_numpy(neg_link_mask)

    assert ((pos_link_mask * neg_link_mask).sum().item() == 0)
    assert (neg_link_mask.sum().item() == num_links)
    assert (((pos_link_mask + neg_link_mask) * test_mask != (pos_link_mask + neg_link_mask)).sum().item() == 0)
    return pos_link_mask + neg_link_mask
