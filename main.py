import warnings
import os, sys

import torch
from tqdm import trange
from torch_geometric.utils import to_networkx, k_hop_subgraph
from datasets import DataLoader
from utils import *
from torch_geometric.datasets import Planetoid
from NHGCN import NHGCN
import argparse
import nni

from config import Config, seed_everything


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    return test_acc


@torch.no_grad()
def run_full_data(data, forcing=True):
    mask = data.train_mask
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1,keepdim=True)  # Use the class with highest probability.
    if forcing:
        pred = ((data.y.detach() + 1) * mask).view(-1, 1) * mask + (pred + 1) * ~mask
        onehot = torch.zeros((out.shape[0], out.shape[1] + 1), device=Config.device)
        onehot.scatter_(1, pred, 1)
        onehot = onehot[:, 1:]
    else:    #return onehot
        onehot = torch.zeros(out.shape, device=Config.device)
        onehot.scatter_(1, pred, 1)
    return onehot


@torch.no_grad()
def valid(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc



if __name__ == "__main__":
    # PARSER BLOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='cora')
    parser.add_argument('--baseseed', '-S', type=int, default=42)
    parser.add_argument('--hidden', '-H', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--finaldp', type=float, default=0.0)
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'tanh'])
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--includeself', '-I', type=int, default=0, choices=[0, 1])
    parser.add_argument('--addself', '-A', type=int, default=1, choices=[0, 1])
    parser.add_argument('--finalagg', '-F', type=str, default='add')
    parser.add_argument('--model', '-M', type=str, default='NHGCN')
    parser.add_argument('--threshold', '-T', type=float, default=2.)
    args = parser.parse_args()
    dataset, data = DataLoader(args.dataset)
    print(f"load {args.dataset} successfully!")
    print('==============================================================')

    warnings.filterwarnings("ignore")

#     optimized_params = nni.get_next_parameter()
    args_dict = vars(args)
#     args_dict.update(optimized_params)
    args = argparse.Namespace(**args_dict)

    train_rate = 0.6
    val_rate = 0.2
    # class balance for training dataset
    num_nodes = dataset.num_nodes
    percls_trn = int(round(train_rate * num_nodes / dataset.num_classes))
    val_lb = int(round(val_rate * num_nodes))
    accs, test_accs = [], []

    # 10 times rand part
    neigh_dict = cal_nei_index(args.dataset,data.edge_index, args.hops, dataset.num_nodes,args.includeself)
    print('indexing finished')
    for rand in trange(10):
        # training settings
        seed_everything(args.baseseed + rand)
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb).to(Config.device)
        if args.model == 'NHGCN':
            model = NHGCN(dataset.num_features, dataset.num_classes, args)
        elif args.model == 'GCN':
            model = GCN_Net(dataset.num_features, dataset.num_classes, args)
        # print(f"init model {args.model} successfully")
        criterion = torch.nn.CrossEntropyLoss()

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # init_cc
        data.cc_mask = torch.ones_like(data.y).float()
        data.update_cc = True
        data, model = data.to(Config.device), model.to(Config.device)

        best_acc = 0.
        final_test_acc = 0.
        es_count = patience = 100
        for epoch in range(500):
            loss, out = train(data)
            data.update_cc = False
            val_acc = valid(data)
            test_acc = test(data)
            if val_acc > best_acc:
                es_count = patience
                best_acc = val_acc
                final_test_acc = test_acc
                predict = run_full_data(data)
                data.cc_mask = cal_cc(neigh_dict, predict.detach().cpu(), args.threshold)
                data.update_cc = True
            else:
                es_count -= 1
            if es_count <= 0:
                break
        accs.append(best_acc)
        test_accs.append(final_test_acc)
    accs = torch.tensor(accs)
    test_accs = torch.tensor(test_accs)
    # nni.report_final_result(accs.mean().item())
    print(f'{args.dataset} valid_acc: {100 * accs.mean().item():.2f} ± {100 * accs.std().item():.2f}')
    print(f'{args.dataset} test_acc: {100 * test_accs.mean().item():.2f} ± {100 * test_accs.std().item():.2f}')
