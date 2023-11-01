import torch
import numpy as np
import math
import torch.nn as nn
from utils import edge_index_to_adj
import sys
import pickle as pkl
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances as pair
from dataset_loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity as cos
def similarity_CN(dataset, data, alpha):
    edge_index = data.edge_index

    adj = edge_index_to_adj(edge_index)# 无自环
    A2 = torch.mm(adj,adj)
    #sim = A2
    feature = data.x
    feature = torch.mm(adj, feature)
    fea_dist = cos(feature.numpy())  # 邻域特征之间的相似性
    fea_dist = torch.from_numpy(fea_dist)

    sim = alpha*A2 + (1-alpha)*fea_dist


    return sim
##################################################
def construct_struct_graph(dataset, sim, topk, alpha):
    fname = './CN_data/' + dataset +'/' + str(alpha) + '/tmpe.txt'
    print(fname)
    f = open(fname, 'w')

    #### CN similarity #############
    dist = sim
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass #去除自环
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_struct(dataset,sim,alpha):
    for topk in range(1, 6):
        construct_struct_graph(dataset, sim, topk,alpha)
        f1 = open('./CN_data/' + dataset +'/' + str(alpha) + '/tmpe.txt','r')
        f2 = open('./CN_data/' + dataset +'/' + str(alpha) + '/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()
