import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits, load_graph, set_seed
from model import UGCN_SN
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time

import pandas as pd


def RunExp(args, dataset, data, Net, percls_trn, val_lb, seed):
    def train(model, optimizer, data, adj, sadj):
        model.train()
        optimizer.zero_grad()
        out = model(data, adj, sadj)
        nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss = nll
        loss.backward()
        optimizer.step()
        del out

    def test(model, data, adj, sadj):
        model.eval()
        logits = model(data, adj, sadj)
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')


    # randomly split dataset
    permute_masks = random_planetoid_splits
    data, train_index = permute_masks(data, dataset.num_classes, percls_trn, val_lb, seed)
    adj, sadj = load_graph(data, args)
    adj = adj.to(device)
    sadj = sadj.to(device)

    tmp_net = Net(dataset, args)
    model, data = tmp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data, adj, sadj)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, adj, sadj)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break

    return test_acc, best_val_acc, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.08, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.8, help='dropout for neural networks.')
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')

    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Chameleon', 'Squirrel', 'Actor',
                                 'Texas', 'Cornell', 'Wisconsin'],
                        default='Chameleon')
    parser.add_argument('--device', type=int, default=3, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')

    parser.add_argument('--ks', type=int, default=3, help='number of runs.')

    parser.add_argument('--net', type=str, default='UGCN_SN')

    args = parser.parse_args()
    # set_seed(args.seed)
    # 10 fixed seeds for splits
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    Net = UGCN_SN

    dataset = DataLoader(args.dataset)
    data = dataset[0]

    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    results = []
    time_results = []
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        set_seed(args.seed)
        test_acc, best_val_acc, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb, args.seed)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values = np.asarray(results)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    f = open("./res/{}.txt".format(args.dataset), 'a')
    f.write(
        "lr:{}, wd:{} ,hid:{}, drop:{}, acc_test:{},std:{},time1:{},time2:{},ks:{}".format(args.lr, args.weight_decay,
                                                                                           args.hidden, args.dropout,
                                                                                           test_acc_mean,
                                                                                           uncertainty * 100,
                                                                                           run_sum / (args.runs),
                                                                                           1000 * run_sum / epochsss,
                                                                                           args.ks))
    f.write("\n")
    f.close()
