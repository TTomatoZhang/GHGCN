from __future__ import print_function, division
import math
import numpy as np
import torch
from components import *
import sys
from utils import *
from layers import *
#3
def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    """

    :param logits: logits = torc.clamp(logits, min=-10, max=10
    :param target: label
    :param pos_weight:
    :return:
    """
    x = logits
    z = targets
    l = 1 + (pos_weight - 1) * targets

    loss = (l - z) * x + l * (torch.log(1 + torch.exp(-torch.abs(x))) + torch.clamp(-x, min=0))
    return loss
#2
def KL_normal(z_mean_1, z_std_1, z_mean_2, z_std_2):
    kl = torch.log(z_std_2 / z_std_1) + ((z_std_1 ** 2) + (z_mean_1 - z_mean_2) ** 2) / (2 * z_std_2 ** 2) - 0.5
    return kl.sum(1).mean()

# def reconstruction_loss(adj, adj_h, mask=None, test=False, fixed_norm=None):
#     if not test:
#         norm = adj.shape[0] ** 2 / float((adj.shape[0] ** 2 - adj.sum()) * 2)
#         pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     else:
#         norm = 1.0
#         pos_weight = 1.0
#
#     if fixed_norm is not None:
#         norm = fixed_norm
#         pos_weight = 1.0
#
#2
def reconstruction_loss_A(adj, adj_h,mask=None, test=False, fixed_norm=None):
    if not test:
        norm = adj.shape[0] ** 2 / float((adj.shape[0] ** 2 - adj.sum()) * 2)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    else:
        norm = 1.0
        pos_weight = 1.0

    if fixed_norm is not None:
        norm = fixed_norm
        pos_weight = 1.0


    element_loss = weighted_cross_entropy_with_logits(adj_h, adj, pos_weight)
    if mask is not None:
        element_loss = element_loss[mask]
    neg_log_lik = norm * torch.mean(element_loss)
    return neg_log_lik

#1
def vae_loss_classify(z_mean, z_log_std, adj, X, label,  classifier, fixed_norm=None):
    adj_h = sample_reconstruction(z_mean, z_log_std)
    #H_h = G2H(adj_h)# this step needs not gradient
    #neg_log_lik = reconstruction_loss_H(H, H_h, fixed_norm=fixed_norm)
    neg_log_lik = reconstruction_loss_A(adj_h, adj, fixed_norm=fixed_norm) #TODO why there is a norm?
    z_std = torch.exp(z_log_std)
    kl = KL_normal(z_mean, z_std, 0.0, 1.0)
    Z = classifier(adj, X)
    label = torch.squeeze(label)
    #loss_c = torch.nn.NLLLoss()
    #loss_classify = loss_c(Z, label)
    # for i in range(label.shape[0]):
    #     loss_classify += F.nll_loss(Z[i, :], label[i, :])
    loss_classify = F.nll_loss(Z, label)
    return loss_classify + neg_log_lik + kl / z_mean.size()[0]

#1
def vae_loss_predict(z_mean, z_log_std, adj, X, label,  predictor, fixed_norm=None):
    adj_h = sample_reconstruction(z_mean, z_log_std)
    #H_h = G2H(adj_h)# this step needs not gradient
    #neg_log_lik = reconstruction_loss_H(H, H_h, fixed_norm=fixed_norm)
    neg_log_lik = reconstruction_loss_A(adj_h, adj, fixed_norm) #TODO why there is a norm?
    z_std = torch.exp(z_log_std)
    kl = KL_normal(z_mean, z_std, 0.0, 1.0)
    z = predictor(adj, X)
    loss_predict = F.mse_loss(z, label)
    return loss_predict + neg_log_lik + kl / z_mean.size()[0]

# def r_vae_loss_a(z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj):
#     z_mean = torch.cat((z_mean_old, z_mean_new))
#     z_log_std = torch.cat((z_log_std_old, z_log_std_new))
#
#     adj_h = sample_reconstruction(z_mean, z_log_std)
#     loss = reconstruction_loss_A(adj, adj_h)
#     z_std_new = torch.exp(z_log_std_new)
#     kl = KL_normal(z_mean_new, z_std_new, 0.0, 1.0)
#     # kl = torch.mean(torch.log(1 / z_std_new) + (z_std_new ** 2 + (z_mean_new - 0) ** 2) * 0.5)
#     loss += kl * (z_mean_new.size()[0] / z_mean.size()[0] ** 2)

#    return loss
# no
# def r_vae_loss(z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, H):
#     z_mean = torch.cat((z_mean_old, z_mean_new))
#     z_log_std = torch.cat((z_log_std_old, z_log_std_new))
#     adj_h = sample_reconstruction(z_mean, z_log_std)
#     H_h = G2H(adj_h)
#     loss = reconstruction_loss_A(H, H_h)
#     z_std_new = torch.exp(z_log_std_new)
#     kl = KL_normal(z_mean_new, z_std_new, 0.0, 1.0)
#     loss += kl * (z_mean_new.size()[0] / z_mean.size()[0] ** 2)
#
#     return loss

def r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new, z_log_std_new, adj):
    z_mean = torch.cat((z_mean_old, z_mean_new))
    z_log_std = torch.cat((z_log_std_old, z_log_std_new))

    adj_h = sample_reconstruction(z_mean, z_log_std)
    lossA = reconstruction_loss_A(adj_h, adj)
    #H_h = G2H(adj_h)
    loss = reconstruction_loss_A(adj, adj_h)
    last_z_std = torch.exp(last_z_log_std)
    z_std_old = torch.exp(z_log_std_old)

    kl_last = KL_normal(z_mean_old, z_std_old, last_z_mean, last_z_std)
    kl_last *= (z_mean_old.size()[0] / z_mean.size()[0] ** 2)

    z_std_new = torch.exp(z_log_std_new)
    kl_new = KL_normal(z_mean_new, z_std_new, 0.0, 1.0)
    kl_new *= (1.0 / z_mean.size()[0] ** 2)# nan
    loss += 10 * kl_last
    loss += 10 * kl_new
    return loss


def recursive_loss_with_noise(gcn_step, adj, feat, labels, size_update, classifier, fixed_norm=1.2):
    labels = torch.squeeze(labels)
    num_step = int(math.ceil(1.0 * adj.size()[0]))

    last_z_mean = None
    last_z_log_std = None
    for step in range(num_step):
        if step == 0:
            #H_feed = H[:size_update, :]#
            adj_feed = adj[:size_update, :size_update]
            adj_feed = anormalize(adj_feed)
            feat = torch.squeeze(feat)
            feat_feed = feat[:size_update, :]
            labels = torch.squeeze(labels)
            labels_feed = labels[:size_update]
            z_mean, z_log_std = gcn_step(adj_feed, feat_feed)

            #H_truth = H[:size_update, :]#TODO
            adj_truth = adj[:size_update, :size_update]
            adj_truth = anormalize(adj_truth)
            loss = vae_loss_classify(z_mean, z_log_std, adj_truth, feat_feed, labels_feed, classifier, fixed_norm=fixed_norm)
            # z_mean, z_log_std, adj, X, label,  classifier, fixed_norm=None
            last_z_mean, last_z_log_std = z_mean, z_log_std
            continue

        start_idx = step * size_update
        end_idx = min(adj.size()[0], start_idx + size_update)
        if start_idx == end_idx:
            print('all the hyperedges added')
            sys.exit(0)
        #H_feed = H[:start_idx, :]
        adj_feed = adj[:start_idx, :start_idx]
        adj_feed = anormalize(adj_feed)

        feat_feed = feat[:start_idx, :]
        feat_new = feat[start_idx:end_idx, :]
        try:
            assert feat_new.size()[0] > 0
        except:
            p = 1
        z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed, feat_feed, feat_new)
        # H_truth = H[:end_idx, :end_idx]
        adj_truth = adj[:end_idx, :end_idx]
        # try:
        adj_truth = anormalize(adj_truth)
        # except:
            #p = 1

        curr_loss = r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new,
                                     z_log_std_new, adj_truth)
        loss += curr_loss

        # updating hidden space
        #H_feed = H[:start_idx, :]
        adj_feed = adj[:end_idx, :end_idx]
        adj_feed = anormalize(adj_feed)

        feat_feed = feat[:end_idx, :]
        last_z_mean, last_z_log_std = gcn_step(adj_feed, feat_feed)

    return loss / num_step

# def recursive_loss_with_noise_predict(gcn_step, H, feat, size_update, fixed_norm=1.2):
#     num_step = int(math.ceil(1.0 * H.size()[0]))
#
#     last_z_mean = None
#     last_z_log_std = None
#     for step in range(num_step):
#         if step == 0:
#             H_feed = H[:size_update, :]#TODO
#             adj_feed = H2G(H_feed)
#             adj_feed = anormalize(adj_feed)
#             feat_feed = feat[:size_update, :]
#             z_mean, z_log_std = gcn_step(adj_feed, feat_feed)
#
#             H_truth = H[:size_update, :]#TODO
#             adj_truth = H2G(H_truth)
#             adj_truth = anormalize(adj_truth)
#             loss = vae_loss_predict(z_mean, z_log_std, adj_truth, fixed_norm=fixed_norm)
#             last_z_mean, last_z_log_std = z_mean, z_log_std
#             continue
#
#             start_idx = step * size_update
#             end_idx = min(adj.size()[0], start_idx + size_update)
#             H_feed = H[:start_idx, :]
#             adj_feed = H2G(H_feed)
#             adj_feed = anormalize(adj_feed)
#
#             feat_feed = feat[:start_idx, :]
#             fead_new = feat[start_idx:end_idx, :]
#             z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed, feat_feed, fead_new)
#             H_truth = H[:end_idx, :end]
#             adj_truth = H2G(H_truth)
#             adj_truth = anormalize(adj_truth)
#
#             curr_loss = r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new,
#                                          z_log_std_new, adj_truth, fixed_norm=fixed_norm)
#             loss += curr_loss * end_idx ** 2
#
#             # updating hidden space
#             H_feed = H[:start_idx, :]
#             adj_feed = H2G(H_feed)
#             adj_feed = anormalize(adj_feed)
#
#             feat_feed = feat[:start_idx, :]
#             last_z_mean, last_z_log_std = gcn_step(adj_feed, feat_feed)
#
#             return loss / num_step


#no
# def recursive_loss_with_noise_supervised(gcn_step, H, label, feat, size_update, fixed_norm=1.2):
#     num_step = int(math.ceil(H.size()[0] / size_update))
#
#     # print("num step: {}".format(num_step))
#
#     last_z_mean = None
#     last_z_log_std = None
#     for step in range(num_step - 1):
#         if step == 0:
#             adj_feed = torch.eye(size_update)
#             feat_feed = feat[:size_update, :]
#             z_mean, z_log_std = gcn_step(adj_feed, feat_feed)
#
#             label_truth = label[0:size_update, 0:size_update]
#             loss = vae_loss(z_mean, z_log_std, label_truth, fixed_norm=fixed_norm)
#             last_z_mean, last_z_log_std = z_mean, z_log_std
#             continue
#
#         start_idx = step * size_update
#         end_idx = min(H.size()[0], start_idx + size_update)
#         H_feed = H[:start_idx, :]
#         adj_feed = H2G(H_feed)
#         adj_feed = anormalize(adj_feed)
#
#         feat_feed = feat[:start_idx, :]
#         fead_new = feat[start_idx:end_idx, :]
#         z_mean_old, z_log_std_old, z_mean_new, z_log_std_new = gcn_step(adj_feed, feat_feed, fead_new)
#         label_truth = label[:end_idx, :end_idx]
#
#         curr_loss = r_vae_loss_addon(last_z_mean, last_z_log_std, z_mean_old, z_log_std_old, z_mean_new, z_log_std_new,
#                                      label_truth, fixed_norm=fixed_norm)
#         loss += curr_loss * (end_idx + 1) ** 2
#
#         # update hidden latent spaces
#         last_z_mean = torch.cat((z_mean_old, z_mean_new))
#         last_z_log_std = torch.cat((z_log_std_old, z_log_std_new))
#     return loss / num_step



