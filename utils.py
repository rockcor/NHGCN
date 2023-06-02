#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
from tqdm import trange
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import os
from config import Config


def cal_nei_index(name, ei, k, num_nodes, include_self=1):
    if include_self:
        path_name = f'index/{name}_hop{k}.npy'
    else:
        path_name = f'index/{name}_hop{k}_noself.npy'
    if os.path.exists(path_name):
        neigh_dict = np.load(path_name, allow_pickle=True).item()
    else:
        neigh_dict = {}
        for id in trange(num_nodes):
            # neigh = k_hop_subgraph(id, k, ei)[0]
            # exclude self
            if include_self:
                neigh = k_hop_subgraph(id, k, ei)[0]
            else:
                neigh = k_hop_subgraph(id, k, ei)[0][1:]
            neigh_dict[id] = neigh
        np.save(path_name, neigh_dict)
    return neigh_dict


# bounded with cal_nei_index
def cal_hn(nei_dict, y, thres=0.5, soft=False):
    hn = np.empty(len(y), dtype=float)
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        labels = labels[labels == y[i]]
        if len(neigh):
            hn[i] = len(labels) / len(neigh)
        else:
            hn[i] = 1

    if soft:
        return hn
    mask = np.where(hn <= thres, 1., 0.)
    return torch.from_numpy(mask).float().to(Config.device)



def cal_cc(nei_dict, y, thres=2., use_tensor=True, soft=False):
    cc = np.empty(y.shape[0])
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        if len(labels):
            cc[i] = len(labels) / torch.max(torch.sum(labels, dim=0)).item()
        else:
            cc[i] = 1.0

    if soft:
        return cc
    # low_cc: 1 ; high_cc: 0
    mask = np.where(cc <= thres, 1., 0.)
    if use_tensor:
        return torch.from_numpy(mask).float().to(Config.device)
    else:
        return mask


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    data.test_mask = index_to_mask(
        rest_index[val_lb:], size=data.num_nodes)

    return data
