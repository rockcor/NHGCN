#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import numpy as np

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor, LINKXDataset
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from torch_geometric.io import read_npz


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    url = ('https://gitee.com/rockcor/geom-gcn/tree/master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class NPZ:
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        path = osp.join(root, name + '.npz')
        if self.name in ['cora_full','cocs','cophy']:
            data = read_npz(path)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])
            edge_index, _ = coalesce(edge_index, None, data.x.size(0), data.x.size(0))
            self.data = Data(x=data.x, edge_index=edge_index, y=data.y)
        else:
            self.data = self.read_other_npz(path)

        self.data = self.data if pre_transform is None else pre_transform(self.data)
        self.num_classes = len(self.data.y.unique())
        self.num_features = self.data.x.shape[1]
        self.num_nodes = len(self.data.y)

    def read_other_npz(self, path):
        with np.load(path) as f:
            x = torch.from_numpy(f['node_features']).to(torch.float)
            edge_index = torch.from_numpy(f['edges'].T).to(torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            y = torch.from_numpy(f['node_labels']).to(torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


def DataLoader(name):
    if name in ['cora_full', 'wiki_cooc', 'questions', 'roman_empire', 'amazon_ratings', 'squirrel_filtered',
                'chameleon_filtered', 'actor','cocs','cophy']:
        root_path = '../data'
        dataset = NPZ(root_path, name, pre_transform=T.NormalizeFeatures())
        return dataset, dataset.data
    elif name in ['penn94','genius']:
        root_path = '../data/'
        dataset = LINKXDataset(root_path,name, pre_transform=T.NormalizeFeatures())
        dataset.num_nodes = len(dataset[0].y)
        return dataset, dataset.data

    elif name in ['cora', 'citeseer', 'pubmed']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        root_path = '../'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        edge_index, _ = remove_self_loops(preProcDs[0].edge_index)
        edge_index = to_undirected(edge_index, num_nodes=data.x.size(0))
        data.edge_index, _ = coalesce(edge_index, None, data.x.size(0), data.x.size(0))
        dataset.num_nodes = len(data.y)
        return dataset, data

    elif name in ['film']:
        dataset = Actor(
            root='../data/film',transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    dataset.num_nodes = len(dataset[0].y)
    return dataset, dataset[0]
