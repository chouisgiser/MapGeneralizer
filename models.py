# -*- coding: utf-8 -*-
"""
# @time    : 05.05.22 14:11
# @author  : zhouzy
# @file    : models.py
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, GraphNorm, BatchNorm, LayerNorm, GraphSizeNorm, InstanceNorm, TopKPooling
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj
torch.Generator().manual_seed(0)
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NodeClsGCN(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dropout):
        super(NodeClsGCN, self).__init__()
        self.gc1 = GCNConv(in_channels, hidden_dims[0])
        self.gc2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.gc3 = GCNConv(hidden_dims[1], out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, edge_index)
        return F.log_softmax(x, dim=1)

class NodeClsGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dropout=None, task=None):
        super(NodeClsGraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
        self.rm_norm1 = BatchNorm(hidden_dims[0])

        self.sage2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
        self.rm_norm2 = BatchNorm(hidden_dims[1])

        self.sage4 = SAGEConv(hidden_dims[1], out_channels, aggr='min')

        self.dropout = dropout
        self.task = task

    def forward(self, x, edge_index):
        x = F.relu(self.rm_norm1(self.sage1(x, edge_index)))
        x = F.relu(self.rm_norm2(self.sage2(x, edge_index)))
        x = self.sage4(x, edge_index)

        return F.log_softmax(x, dim=1)

class NodeClsGAT(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dropout):
        super(NodeClsGAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_dims[0])
        self.gat2 = GATConv(hidden_dims[0], hidden_dims[1])
        self.gat3 = GATConv(hidden_dims[1], out_channels)


    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)

class NodeRegGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, dropout=None, task=None):
        super(NodeRegGraphSAGE, self).__init__()

        self.sage1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
        self.rm_norm1 = BatchNorm(hidden_dims[0])

        self.sage2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
        self.rm_norm2 = BatchNorm(hidden_dims[1])

        self.fc1 = Linear(hidden_dims[1], out_channels)
        self.dropout = dropout
        self.task = task

    def forward(self, x, edge_index):
        x = F.relu(self.rm_norm1(self.sage1(x, edge_index)))

        x = F.relu(self.rm_norm2(self.sage2(x, edge_index)))

        x = self.fc1(x)
        return torch.flatten(x)


# normalization layer
class BuildingGenModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None, model='GraphSAGE'):
        super(BuildingGenModel, self).__init__()
        # self.dropout = dropout
        if model == 'GraphSAGE':
            self.rm_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
            self.rm_norm1 = BatchNorm(hidden_dims[0])
            self.rm_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rm_norm2 = BatchNorm(hidden_dims[1])
            self.rm_l4 = SAGEConv(hidden_dims[1], 3, aggr='min')
            # self.rm_l4 = Linear(hidden_dims[1], 3)

            self.shared_l1 = SAGEConv(in_channels + 3, hidden_dims[0], aggr='min')
            self.shared_norm1 = BatchNorm(hidden_dims[0])

            self.premove_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.premove_norm1 = BatchNorm(hidden_dims[1])
            self.premove_l3 = Linear(hidden_dims[1], 1)
            # self.rtang_l3 = SAGEConv(hidden_dims[1], 1, aggr='sum')

            self.sucmove_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.sucmove_norm1 = BatchNorm(hidden_dims[1])
            self.sucmove_l3 = Linear(hidden_dims[1], 1)
            # self.movedis_l3 = SAGEConv(hidden_dims[1], 1, aggr='sum')

            self.weights = torch.nn.Parameter(torch.ones(4).float())
        elif model == 'GCN':
            self.rm_l1 = GCNConv(in_channels, hidden_dims[0], aggr='min')
            self.rm_norm1 = BatchNorm(hidden_dims[0])
            self.rm_l2 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rm_norm2 = BatchNorm(hidden_dims[1])
            self.rm_l4 = GCNConv(hidden_dims[1], 3, aggr='min')

            self.shared_l1 = GCNConv(in_channels + 3, hidden_dims[0], aggr='min')
            self.shared_norm1 = BatchNorm(hidden_dims[0])

            self.premove_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.premove_norm1 = BatchNorm(hidden_dims[1])
            self.premove_l3 = Linear(hidden_dims[1], 1)

            self.sucmove_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.sucmove_norm1 = BatchNorm(hidden_dims[1])
            self.sucmove_l3 = Linear(hidden_dims[1], 1)

            self.weights = torch.nn.Parameter(torch.ones(4).float())
        elif model == 'GAT':
            self.rm_l1 = GATConv(in_channels, hidden_dims[0], aggr='min')
            self.rm_norm1 = BatchNorm(hidden_dims[0])
            self.rm_l2 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rm_norm2 = BatchNorm(hidden_dims[1])
            self.rm_l4 = GATConv(hidden_dims[1], 3, aggr='min')

            self.shared_l1 = GATConv(in_channels + 3, hidden_dims[0], aggr='min')
            self.shared_norm1 = BatchNorm(hidden_dims[0])

            self.premove_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.premove_norm1 = BatchNorm(hidden_dims[1])
            self.premove_l3 = Linear(hidden_dims[1], 1)

            self.sucmove_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.sucmove_norm1 = BatchNorm(hidden_dims[1])
            self.sucmove_l3 = Linear(hidden_dims[1], 1)

            self.weights = torch.nn.Parameter(torch.ones(4).float())
        else:
            print("Invalid graph convlutional operation.")

    def forward(self, x, edge_index):

        rm = F.relu(self.rm_norm1(self.rm_l1(x, edge_index)))

        rm = F.relu(self.rm_norm2(self.rm_l2(rm, edge_index)))

        rm = self.rm_l4(rm, edge_index)

        x = F.relu(self.shared_norm1(self.shared_l1(torch.cat((x, rm), 1), edge_index)))

        rm = F.log_softmax(rm, dim=1)

        rm_labels = rm.max(1)[1]
        rm_labels = torch.where(rm_labels == 2, 1, 0)

        preMove = F.relu(self.premove_norm1(self.premove_l1(x, edge_index)))

        preMove = self.premove_l3(preMove)
        preMove = torch.flatten(preMove)
        preMove = torch.mul(preMove, rm_labels)

        sucMove = F.relu(self.sucmove_norm1(self.sucmove_l1(x, edge_index)))

        sucMove = self.sucmove_l3(sucMove)
        sucMove = torch.flatten(sucMove)
        sucMove = torch.mul(sucMove, rm_labels)

        return rm, preMove, sucMove
