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

class GCAEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super(GCAEncoder, self).__init__()
        self.gc1 = GCNConv(in_channels, hidden_dims[0])
        self.gc2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.gc3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.pooling_signals = list()

    def forward(self, x, edge_index):
        self.pooling_signals.append(edge_index)
        x = F.relu(self.gc1(x, edge_index))
        x, edge_index = self.graph_max_pool(x, edge_index)

        x = F.relu(self.gc2(x, edge_index))
        self.pooling_signals.append(edge_index)
        x, edge_index = self.graph_max_pool(x, edge_index)

        x = F.relu(self.gc3(x, edge_index))
        self.pooling_signals.append(edge_index)
        latent, edge_index = self.graph_max_pool(x, edge_index)

        return latent, edge_index

    def graph_max_pool(self, x, edge_index):
        x = nn.functional.max_pool1d(x.permute(1, 0), kernel_size=2).permute(1, 0)
        out_dims = x.shape[0]
        source_list = list()
        target_list = list()
        for i in range(edge_index.shape[1]):
            if edge_index[0][i] < out_dims and edge_index[1][i] < out_dims:
                source_list.append(edge_index[0][i])
                target_list.append(edge_index[1][i])
        source_list.extend([out_dims-1, 0])
        target_list.extend([0, out_dims-1])
        edge_index = torch.tensor([source_list, target_list])
        return x.to(device), edge_index.to(device)

class GCADecoder(nn.Module):
    def __init__(self, hidden_dims, out_channels):
        super(GCADecoder, self).__init__()
        self.gc1 = GCNConv(hidden_dims[2], hidden_dims[2])
        self.gc2 = GCNConv(hidden_dims[2], hidden_dims[1])
        self.gc3 = GCNConv(hidden_dims[1], hidden_dims[0])
        self.gc4 = GCNConv(hidden_dims[0], out_channels)

    def forward(self, z, edge_index, sigmoid=True):
        z = F.relu(self.gc1(z, edge_index))
        z = self.graph_upsamping(z)

        z = F.relu(self.gc2(z, self.pooling_signals[2]))
        z = self.graph_upsamping(z)

        z = F.relu(self.gc3(z, self.pooling_signals[1]))
        z = self.graph_upsamping(z)

        z = self.gc4(z, self.pooling_signals[0])

        return z

        # adj = torch.matmul(z, z.t())
        # return torch.sigmoid(adj) if sigmoid else adj

    def graph_upsamping(self, z):
        z = torch.unsqueeze(z, 0)
        z = nn.functional.interpolate(z.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        z = torch.squeeze(z)
        return z.to(device)

    def get_pooling_signals(self, encoder):
        self.pooling_signals = encoder.pooling_signals

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
            
            self.rtang_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rtang_norm1 = BatchNorm(hidden_dims[1])
            self.rtang_l3 = Linear(hidden_dims[1], 1)
            # self.rtang_l3 = SAGEConv(hidden_dims[1], 1, aggr='sum')
            
            self.movedis_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.movedis_norm1 = BatchNorm(hidden_dims[1])
            self.movedis_l3 = Linear(hidden_dims[1], 1)
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
            
            self.rtang_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rtang_norm1 = BatchNorm(hidden_dims[1])
            self.rtang_l3 = Linear(hidden_dims[1], 1)
            
            self.movedis_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.movedis_norm1 = BatchNorm(hidden_dims[1])
            self.movedis_l3 = Linear(hidden_dims[1], 1)
            
            self.weights = torch.nn.Parameter(torch.ones(4).float())
        elif model == 'GAT':
            self.rm_l1 = GATConv(in_channels, hidden_dims[0], aggr='min')
            self.rm_norm1 = BatchNorm(hidden_dims[0])
            self.rm_l2 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rm_norm2 = BatchNorm(hidden_dims[1])
            self.rm_l4 = GATConv(hidden_dims[1], 3, aggr='min')

            self.shared_l1 = GATConv(in_channels + 3, hidden_dims[0], aggr='min')
            self.shared_norm1 = BatchNorm(hidden_dims[0])
            
            self.rtang_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.rtang_norm1 = BatchNorm(hidden_dims[1])
            self.rtang_l3 = Linear(hidden_dims[1], 1)
            
            self.movedis_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
            self.movedis_norm1 = BatchNorm(hidden_dims[1])
            self.movedis_l3 = Linear(hidden_dims[1], 1)
            
            self.weights = torch.nn.Parameter(torch.ones(4).float())
        else:
            print("Invalid graph convlutional operation.")

    def forward(self, x, edge_index):

        rm = F.relu(self.rm_norm1(self.rm_l1(x, edge_index)))
        # rm = F.relu(self.rm_l1(x, edge_index))
        # rm = F.dropout(rm, self.dropout, training=self.training)
        
        rm = F.relu(self.rm_norm2(self.rm_l2(rm, edge_index)))
        # rm = F.relu(self.rm_l2(rm, edge_index))
        # rm = F.dropout(rm, self.dropout, training=self.training)

        rm = self.rm_l4(rm, edge_index)
        # rm = self.rm_l4(rm)

        x = F.relu(self.shared_norm1(self.shared_l1(torch.cat((x, rm), 1), edge_index)))
        # x = F.relu(self.shared_l1(torch.cat((x, rm), 1), edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.shared_norm1(self.shared_l1(x, edge_index)))
        
        rm = F.log_softmax(rm, dim=1)
        
        rm_labels = rm.max(1)[1]
        rm_labels = torch.where(rm_labels == 2, 1, 0)
        # rm_labels = torch.where(rm_labels == 0, 0, 1)
        
        rtAngle = F.relu(self.rtang_norm1(self.rtang_l1(x, edge_index)))
        # rtAngle = F.relu(self.rtang_l1(x, edge_index))
        # rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        # rtAngle = F.relu(self.rtang_norm1(self.rtang_l1(x)))
        rtAngle = self.rtang_l3(rtAngle)
        rtAngle = torch.flatten(rtAngle)
        rtAngle = torch.mul(rtAngle, rm_labels)

        moveDis = F.relu(self.movedis_norm1(self.movedis_l1(x, edge_index)))
        # moveDis = F.relu(self.movedis_l1(x, edge_index))
        # moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        # moveDis = F.relu(self.movedis_norm1(self.movedis_l1(x)))
        moveDis = self.movedis_l3(moveDis)
        moveDis = torch.flatten(moveDis)
        moveDis = torch.mul(moveDis, rm_labels)

        return rm, rtAngle, moveDis

# class BuildingGenModel(nn.Module):
#     def __init__(self, in_channels, hidden_dims, dropout=None, model='GraphSAGE'):
#         super(BuildingGenModel, self).__init__()
#         self.dropout = dropout
#         if model == 'GraphSAGE':
#             self.rm_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
#             self.rm_norm1 = BatchNorm(hidden_dims[0])
#             self.rm_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rm_norm2 = BatchNorm(hidden_dims[1])
#             self.rm_l4 = SAGEConv(hidden_dims[1], 3, aggr='min')
#             # self.rm_l4 = Linear(hidden_dims[1], 3)
        
#             self.shared_l1 = SAGEConv(in_channels + 3, hidden_dims[0], aggr='min')
#             self.shared_norm1 = BatchNorm(hidden_dims[0])
            
#             self.rtang_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rtang_norm1 = BatchNorm(hidden_dims[1])
#             self.rtang_l3 = Linear(hidden_dims[1], 1)
#             # self.rtang_l3 = SAGEConv(hidden_dims[1], 1, aggr='sum')
            
#             self.movedis_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.movedis_norm1 = BatchNorm(hidden_dims[1])
#             self.movedis_l3 = Linear(hidden_dims[1], 1)
#             # self.movedis_l3 = SAGEConv(hidden_dims[1], 1, aggr='sum')
            
#             self.weights = torch.nn.Parameter(torch.ones(4).float())
#         elif model == 'GCN':
#             self.rm_l1 = GCNConv(in_channels, hidden_dims[0], aggr='min')
#             self.rm_norm1 = BatchNorm(hidden_dims[0])
#             self.rm_l2 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rm_norm2 = BatchNorm(hidden_dims[1])
#             self.rm_l4 = GCNConv(hidden_dims[1], 3, aggr='min')

#             self.shared_l1 = GCNConv(in_channels + 3, hidden_dims[0], aggr='min')
#             self.shared_norm1 = BatchNorm(hidden_dims[0])
            
#             self.rtang_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rtang_norm1 = BatchNorm(hidden_dims[1])
#             self.rtang_l3 = Linear(hidden_dims[1], 1)
            
#             self.movedis_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.movedis_norm1 = BatchNorm(hidden_dims[1])
#             self.movedis_l3 = Linear(hidden_dims[1], 1)
            
#             self.weights = torch.nn.Parameter(torch.ones(4).float())
#         elif model == 'GAT':
#             self.rm_l1 = GATConv(in_channels, hidden_dims[0], aggr='min')
#             self.rm_norm1 = BatchNorm(hidden_dims[0])
#             self.rm_l2 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rm_norm2 = BatchNorm(hidden_dims[1])
#             self.rm_l4 = GATConv(hidden_dims[1], 3, aggr='min')

#             self.shared_l1 = GATConv(in_channels + 3, hidden_dims[0], aggr='min')
#             self.shared_norm1 = BatchNorm(hidden_dims[0])
            
#             self.rtang_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.rtang_norm1 = BatchNorm(hidden_dims[1])
#             self.rtang_l3 = Linear(hidden_dims[1], 1)
            
#             self.movedis_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='min')
#             self.movedis_norm1 = BatchNorm(hidden_dims[1])
#             self.movedis_l3 = Linear(hidden_dims[1], 1)
            
#             self.weights = torch.nn.Parameter(torch.ones(4).float())
#         else:
#             print("Invalid graph convlutional operation.")

#     def forward(self, x, edge_index):

#         # rm = F.relu(self.rm_norm1(self.rm_l1(x, edge_index)))
#         rm = F.relu(self.rm_l1(x, edge_index))
#         rm = F.dropout(rm, self.dropout, training=self.training)
        
#         # rm = F.relu(self.rm_norm2(self.rm_l2(rm, edge_index)))
#         rm = F.relu(self.rm_l2(rm, edge_index))
#         # rm = F.dropout(rm, self.dropout, training=self.training)

#         rm = self.rm_l4(rm, edge_index)
#         rm = F.dropout(rm, self.dropout, training=self.training)
#         # rm = self.rm_l4(rm)

#         # x = F.relu(self.shared_norm1(self.shared_l1(torch.cat((x, rm), 1), edge_index)))
#         x = F.relu(self.shared_l1(torch.cat((x, rm), 1), edge_index))
#         x = F.dropout(x, self.dropout, training=self.training)
#         # x = F.relu(self.shared_norm1(self.shared_l1(x, edge_index)))
        
#         rm = F.log_softmax(rm, dim=1)
        
#         rm_labels = rm.max(1)[1]
#         rm_labels = torch.where(rm_labels == 2, 1, 0)
        
#         # rtAngle = F.relu(self.rtang_norm1(self.rtang_l1(x, edge_index)))
#         rtAngle = F.relu(self.rtang_l1(x, edge_index))
#         rtAngle = self.rtang_l3(rtAngle)
#         # rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
#         rtAngle = torch.flatten(rtAngle)
#         rtAngle = torch.mul(rtAngle, rm_labels)

#         # moveDis = F.relu(self.movedis_norm1(self.movedis_l1(x, edge_index)))
#         moveDis = F.relu(self.movedis_l1(x, edge_index))
#         moveDis = self.movedis_l3(moveDis)
#         # moveDis = F.dropout(moveDis, self.dropout, training=self.training)
#         moveDis = torch.flatten(moveDis)
#         moveDis = torch.mul(moveDis, rm_labels)

#         return rm, rtAngle, moveDis

# class BuildingGenModel(nn.Module):
#     def __init__(self, in_channels, hidden_dims, dropout=None):
#         super(BuildingGenModel, self).__init__()
        
#         self.rm_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
#         self.rm_norm1 = BatchNorm(hidden_dims[0])
        
#         self.rm_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         self.rm_norm2 = BatchNorm(hidden_dims[1])
        
#         self.rm_l4 = SAGEConv(hidden_dims[1], 3, aggr='min')
        
#         self.shared_l1 = SAGEConv(in_channels + 3, hidden_dims[0], aggr='min')
#         self.shared_norm1 = BatchNorm(hidden_dims[0])
        
#         self.rtang_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         self.rtang_norm1 = BatchNorm(hidden_dims[1])
        
#         # self.rtang_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         # self.rtang_norm2 = BatchNorm(hidden_dims[1])
        
#         self.rtang_l3 = Linear(hidden_dims[1], 1)
        
#         self.movedis_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         self.movedis_norm1 = BatchNorm(hidden_dims[1])
        
#         # self.movedis_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         # self.movedis_norm2 = BatchNorm(hidden_dims[1])

#         self.movedis_l3 = Linear(hidden_dims[1], 1)

#         self.weights = torch.nn.Parameter(torch.ones(4).float())

#     def forward(self, x, edge_index, batch):

#         rm = F.relu(self.rm_norm1(self.rm_l1(x, edge_index)))
        
#         rm = F.relu(self.rm_norm2(self.rm_l2(rm, edge_index)))

#         rm = self.rm_l4(rm , edge_index)

#         x = F.relu(self.shared_norm1(self.shared_l1(torch.cat((x, rm), 1), edge_index)))
        
#         rm = F.log_softmax(rm, dim=1)
        
#         rm_labels = rm.max(1)[1]
#         rm_labels = torch.where(rm_labels == 2, 1, 0)
#         # rm_labels = torch.where(rm_labels == 0, 0, 1)
        
#         # rtAngle = F.relu(self.rtang_norm1(self.rtang_l1(torch.cat((x, rm), 1), edge_index)))
#         rtAngle = F.relu(self.rtang_norm1(self.rtang_l1(x, edge_index)))
#         rtAngle = self.rtang_l3(rtAngle)
#         rtAngle = torch.flatten(rtAngle)
#         rtAngle = torch.mul(rtAngle, rm_labels)

#         # moveDis = F.relu(self.movedis_norm1(self.movedis_l1(torch.cat((x, rm), 1), edge_index)))
#         moveDis = F.relu(self.movedis_norm1(self.movedis_l1(x, edge_index)))
#         moveDis = self.movedis_l3(moveDis)
#         moveDis = torch.flatten(moveDis)
#         moveDis = torch.mul(moveDis, rm_labels)

#         return rm, rtAngle, moveDis

# class BuildingGenModel(nn.Module):
#     def __init__(self, in_channels, hidden_dims, dropout=None):
#         super(BuildingGenModel, self).__init__()
#         self.dropout = dropout
        
#         self.rm_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='min')
#         self.rm_norm1 = BatchNorm(hidden_dims[0])
        
#         self.rm_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         self.rm_norm2 = BatchNorm(hidden_dims[1])
        
#         self.rm_l4 = SAGEConv(hidden_dims[1], 3, aggr='min')
        
#         self.shared_l1 = SAGEConv(in_channels + 3, hidden_dims[0], aggr='min')
#         self.shared_norm1 = BatchNorm(hidden_dims[0])
        
#         self.shared_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         self.shared_norm2 = BatchNorm(hidden_dims[1])
        
#         self.rtang_l3 = Linear(hidden_dims[1], 1)

#         # self.movedis_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='min')
#         # self.movedis_norm1 = BatchNorm(hidden_dims[1])

#         self.movedis_l3 = Linear(hidden_dims[1], 1)

#         self.weights = torch.nn.Parameter(torch.ones(4).float())

#     def forward(self, x, edge_index, batch):

#         rm = F.relu(self.rm_norm1(self.rm_l1(x, edge_index)))
        
#         rm = F.relu(self.rm_norm2(self.rm_l2(rm, edge_index)))

#         rm = self.rm_l4(rm , edge_index)

#         x = F.relu(self.shared_norm1(self.shared_l1(torch.cat((x, rm), 1), edge_index)))
        
#         rm = F.log_softmax(rm, dim=1)
        
#         rm_labels = rm.max(1)[1]
#         rm_labels = torch.where(rm_labels == 2, 1, 0)
#         # rm_labels = torch.where(rm_labels == 0, 0, 1)
        
#         x = F.relu(self.shared_norm2(self.shared_l2(x, edge_index)))
        
#         rtAngle = self.rtang_l3(x)
#         rtAngle = torch.flatten(rtAngle)
#         rtAngle = torch.mul(rtAngle, rm_labels)

#         # moveDis = F.relu(self.movedis_norm1(self.movedis_l1(x, edge_index)))
#         moveDis = self.movedis_l3(x)
#         moveDis = torch.flatten(moveDis)
#         moveDis = torch.mul(moveDis, rm_labels)

#         return rm, rtAngle, moveDis


class BuildingGenRegModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None):
        super(BuildingGenRegModel, self).__init__()
        self.dropout = dropout
        
        self.shared_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='max')

        self.rtang_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.rtang_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.rtang_l3 = Linear(hidden_dims[2], 1)

        self.movedis_l1 = SAGEConv(hidden_dims[0],hidden_dims[1], aggr='max')
        self.movedis_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.movedis_l3 = Linear(hidden_dims[2], 1)

        self.weights = torch.nn.Parameter(torch.ones(4).float())

    def forward(self, x, edge_index):
        
        mv = x
        mv = F.relu(self.shared_l1(mv, edge_index))
        mv = F.dropout(mv, self.dropout, training=self.training)

        
        rtAngle = F.relu(self.rtang_l1(mv, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = F.relu(self.rtang_l2(rtAngle, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = self.rtang_l3(rtAngle)
        rtAngle = torch.flatten(rtAngle)

        moveDis = F.relu(self.movedis_l1(mv, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = F.relu(self.movedis_l2(moveDis, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = self.movedis_l3(moveDis)
        moveDis = torch.flatten(moveDis)

        return rtAngle, moveDis
    
class BuildingVecMoveModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None):
        super(BuildingVecMoveModel, self).__init__()

        self.shared_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='max')
        self.shared_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.dropout = dropout

        self.rtang_l1 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.rtang_l2 = Linear(hidden_dims[2], 1)

        self.movedis_l1 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.movedis_l2 = Linear(hidden_dims[2], 1)

        self.weights = torch.nn.Parameter(torch.ones(2).float())

    def forward(self, x, edge_index):
        x = F.relu(self.shared_l1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.shared_l2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        shared_layer2 = self.shared_l2

        rtAngle = F.relu(self.rtang_l1(x, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = self.rtang_l2(rtAngle)
        rtAngle = torch.flatten(rtAngle)

        moveDis = F.relu(self.movedis_l1(x, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = self.movedis_l2(moveDis)
        moveDis = torch.flatten(moveDis)

        return shared_layer2, rtAngle, moveDis

class BldgsVecMoveJointModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None):
        super(BldgsVecMoveJointModel, self).__init__()
        self.shared_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='max')
        self.shared_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.dropout = dropout

        self.rtang_l1 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.rtang_l2 = Linear(hidden_dims[2], 1)

        self.movedis_l1 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.movedis_l2 = Linear(hidden_dims[2], 1)

        self.joint_l1 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.joint_l2 = Linear(hidden_dims[2], 2)

        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def forward(self, x, edge_index):
        x = F.relu(self.shared_l1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        shared_layer1 = self.shared_l1

        x = F.relu(self.shared_l2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        shared_layer2 = self.shared_l2

        rtAngle = F.relu(self.rtang_l1(x, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = self.rtang_l2(rtAngle)
        rtAngle = torch.flatten(rtAngle)

        moveDis = F.relu(self.movedis_l1(x, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = self.movedis_l2(moveDis)
        moveDis = torch.flatten(moveDis)

        jointRtMove = F.relu(self.joint_l1(x, edge_index))
        jointRtMove = F.dropout(jointRtMove, self.dropout, training=self.training)
        jointRtMove = self.joint_l2(jointRtMove)
        jointRtMove = torch.flatten(jointRtMove)

        return shared_layer2, rtAngle, moveDis, jointRtMove

class BldgsGenJointModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None):
        super(BldgsGenJointModel, self).__init__()
        self.shared_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='max')
        # self.shared_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.dropout = dropout

        self.rm_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.rm_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.rm_l3 = SAGEConv(hidden_dims[2], 2, aggr='max')

        self.rtang_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.rtang_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.rtang_l3 = Linear(hidden_dims[2], 1)
        
        self.movedis_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.movedis_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.movedis_l3 = Linear(hidden_dims[2], 1)

        self.joint_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
        self.joint_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
        self.joint_l3 = Linear(hidden_dims[2], 2)

        self.weights = torch.nn.Parameter(torch.ones(4).float())

    def forward(self, x, edge_index):
        x = F.relu(self.shared_l1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        shared_layer1 = self.shared_l1

        rm = F.relu(self.rm_l1(x, edge_index))
        rm = F.dropout(rm, self.dropout, training=self.training)
        rm = F.relu(self.rm_l2(rm, edge_index))
        rm = F.dropout(rm, self.dropout, training=self.training)
        rm = self.rm_l3(rm , edge_index)
        rm = F.log_softmax(rm , dim=1)

        rtAngle = F.relu(self.rtang_l1(x, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = F.relu(self.rtang_l2(rtAngle, edge_index))
        rtAngle = F.dropout(rtAngle, self.dropout, training=self.training)
        rtAngle = self.rtang_l3(rtAngle)
        rtAngle = torch.flatten(rtAngle)

        moveDis = F.relu(self.movedis_l1(x, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = F.relu(self.movedis_l2(moveDis, edge_index))
        moveDis = F.dropout(moveDis, self.dropout, training=self.training)
        moveDis = self.movedis_l3(moveDis)
        moveDis = torch.flatten(moveDis)

        jointRtMove = F.relu(self.joint_l1(x, edge_index))
        jointRtMove = F.dropout(jointRtMove, self.dropout, training=self.training)
        jointRtMove = F.relu(self.joint_l2(jointRtMove, edge_index))
        jointRtMove = F.dropout(jointRtMove, self.dropout, training=self.training)
        jointRtMove = self.joint_l3(jointRtMove)
        jointRtMove = torch.flatten(jointRtMove)

        return shared_layer1, rm, rtAngle, moveDis, jointRtMove
        
class BldgsRmJointModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout=None, model='GraphSAGE'):
        super(BldgsRmJointModel, self).__init__()
        if model == 'GraphSAGE':
            self.shared_l1 = SAGEConv(in_channels + 3, hidden_dims[0], aggr='max')
            self.dropout = dropout

            self.rm_l1 = SAGEConv(in_channels, hidden_dims[0], aggr='max')
            self.rm_l2 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.rm_l3 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.rm_l4 = SAGEConv(hidden_dims[2], 3, aggr='max')

            self.joint_l1 = SAGEConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.joint_l2 = SAGEConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.joint_l3 = Linear(hidden_dims[2], 2)
        elif model == 'GCN':
            self.shared_l1 = GCNConv(in_channels + 3, hidden_dims[0], aggr='max')
            self.dropout = dropout

            self.rm_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.rm_l2 = GCNConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.rm_l3 = GCNConv(hidden_dims[2], 3, aggr='max')

            self.joint_l1 = GCNConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.joint_l2 = GCNConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.joint_l3 = Linear(hidden_dims[2], 2)
        elif model == 'GAT':
            self.shared_l1 = GATConv(in_channels, hidden_dims[0], aggr='max')
            self.dropout = dropout

            self.rm_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.rm_l2 = GATConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.rm_l3 = GATConv(hidden_dims[2], 3, aggr='max')

            self.joint_l1 = GATConv(hidden_dims[0], hidden_dims[1], aggr='max')
            self.joint_l2 = GATConv(hidden_dims[1], hidden_dims[2], aggr='max')
            self.joint_l3 = Linear(hidden_dims[2], 2)
        else:
            print("Invalid graph convlutional operation.")
            
        self.weights = torch.nn.Parameter(torch.ones(2).float())

    def forward(self, x, edge_index):
        rm = F.relu(self.rm_l1(x, edge_index))
        rm = F.dropout(rm, self.dropout, training=self.training)
        rm = F.relu(self.rm_l2(rm, edge_index))
        rm = F.dropout(rm, self.dropout, training=self.training)
        rm = F.relu(self.rm_l3(rm, edge_index))
        rm = F.dropout(rm, self.dropout, training=self.training)
        rm = self.rm_l4(rm , edge_index)
        
        x = F.relu(self.shared_l1(torch.cat((x, rm), 1), edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        shared_layer1 = self.shared_l1
        
        rm = F.log_softmax(rm , dim=1)
        rm_labels = rm.max(1)[1]
        rm_labels = torch.where(rm_labels == 2, rm_labels, 0)
        rm_labels = torch.cat((rm_labels, rm_labels), 0)
        
        jointRtMove = F.relu(self.joint_l1(x, edge_index))
        jointRtMove = F.dropout(jointRtMove, self.dropout, training=self.training)
        jointRtMove = F.relu(self.joint_l2(jointRtMove, edge_index))
        jointRtMove = F.dropout(jointRtMove, self.dropout, training=self.training)
        jointRtMove = self.joint_l3(jointRtMove)
        jointRtMove = torch.flatten(jointRtMove)
        jointRtMove = torch.mul(jointRtMove, rm_labels)

        return shared_layer1, rm, jointRtMove
