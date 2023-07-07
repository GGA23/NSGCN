import torch
import  math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class GraphConvolution(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features = in_features, out_features

        self.att_vec_k = Parameter(
                torch.FloatTensor(out_features, out_features))

        self.weight_As, self.weight_A, self.weight_mlp = Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features))

        self.att_vec_v = Parameter(torch.FloatTensor(3, 3))

        self.reset_parameters()

    def reset_parameters(self):

        std_att = 1. / math.sqrt(self.att_vec_k.size(1))
        stdv = 1. / math.sqrt(self.weight_As.size(1))
        std_att_vec = 1. / math.sqrt(self.att_vec_v.size(1))

        self.att_vec_k.data.uniform_(-std_att, std_att)

        self.weight_As.data.uniform_(-stdv, stdv)
        self.weight_A.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.att_vec_v.data.uniform_(-std_att_vec, std_att_vec)

    def Attention(self, output_mlp, output_A, output_As):  #
        tao = 3
        output_stru = output_mlp + output_As + output_A
        K = torch.mean(torch.mm((output_stru), self.att_vec_k), dim=0, keepdim=True)
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat(
            [torch.mm((output_mlp), K.T), torch.mm((output_A), K.T),
             torch.mm((output_As), K.T)], 1)), self.att_vec_v) / tao, 1)

        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]


    def forward(self, inputx , adj, sadj):
        output_mlp = F.relu(torch.mm(inputx, self.weight_mlp))
        output_A = F.relu(torch.spmm(adj, (torch.mm(inputx, self.weight_A))))
        output_As = F.relu(torch.spmm(sadj, (torch.mm(inputx, self.weight_As))))

        alpha_mlp, alpha_A, alpha_As = self.Attention(output_mlp, output_A, output_As)
        emb = alpha_mlp*output_mlp + alpha_A*output_A + alpha_As*output_As

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class UGCN_SN(nn.Module):
    def __init__(self, dataset, args):
        super(UGCN_SN, self).__init__()
        self.gcns = nn.ModuleList()
        self.dropout = args.dropout

        self.gcns.append(GraphConvolution(dataset.num_features, args.hidden))
        self.gcns.append(GraphConvolution(args.hidden, dataset.num_classes))

    def forward(self, data, adj, sadj):
        x = data.x
        x = F.dropout(x, self.dropout, training=self.training)
        fea = (self.gcns[0](x, adj, sadj))
        fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj, sadj)

        return F.log_softmax(fea, dim=1)
