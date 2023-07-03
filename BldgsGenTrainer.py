# -*- coding: utf-8 -*-
"""
# @time    : 13.05.22 14:51
# @author  : zhouzy
# @file    : BldgsGenTrainer.py
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from utils import accuracy, precision, recall, MTL_recontruct_points2, MTL_recontruct_points3, MTL_reconstruct_polygon
from models import NodeClsGraphSAGE, NodeRegGraphSAGE, BuildingGenModel
from scipy.sparse import find
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
torch.manual_seed(0)
import geopandas as gpd
# from focal_loss.focal_loss import FocalLoss
from torch_geometric.utils import dense_to_sparse, degree
from utils import automatic_weight

# loss: autoweight
# tasks: node removal, vector move distance along the preceding and succeeding edges respectively
class MTL_BuildingGen(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.cls_criterion = self.get_cls_loss()
        self.reg_criterion = self.get_reg_loss()
        # self.iou_criterion = IoULoss()
        # self.hdf_criterion = HausdorffDisLoss()
        # self.tf_criterion = TurningFncLoss()
        self.optimizer = self.get_optimizer()

    def get_data_loader(self):
        data_loader = dict()

        train_set = list()
        train_adj = self.data['train_adj_matrix']
        print(len(train_adj))
        for idx in range(len(train_adj)):
            source_array, target_array, _ = find(train_adj[idx])
            train_edge_index = torch.tensor([source_array.tolist(), target_array.tolist()], dtype=torch.long)

            no_movement_indices = np.where((self.data['train_features'][idx][:, -3] == 0) | (self.data['train_features'][idx][:, -3] == 1) )
            self.data['train_features'][idx][no_movement_indices[0], -2] = 0
            self.data['train_features'][idx][no_movement_indices[0], -1] = 0
            features_idx = [0, 1, 2, 3, 4, 5]

            train_y = self.data['train_features'][idx][:, [-3, -2, -1]].astype(np.float32)
            for order in self.params['K_orders']:
                    for feature_idx in self.params['features']:
                        features_idx.append(6 + 4 * (order - 1) + feature_idx)
            train_features = self.data['train_features'][idx][:, features_idx].astype(np.float32)
            train_x = torch.from_numpy(train_features)
            # print(train_x.size())
            train_graph = Data(x=train_x, edge_index=train_edge_index)
            # print(train_edge_index.size())
            train_graph.y = torch.from_numpy(train_y)
            train_set.append(train_graph)
        data_loader['train'] = DataLoader(dataset=train_set, batch_size=self.params['batch_size'], shuffle=True,
                                          drop_last=True)

        valid_set = list()
        valid_adj = self.data['valid_adj_matrix']
        print(len(valid_adj))
        for idx in range(len(valid_adj)):
            source_array, target_array, _ = find(valid_adj[idx])
            valid_edge_index = torch.tensor([source_array.tolist(), target_array.tolist()], dtype=torch.long)

            no_movement_indices = np.where((self.data['valid_features'][idx][:, -3] == 0) | (self.data['valid_features'][idx][:, -3] == 1) )
            self.data['valid_features'][idx][no_movement_indices[0], -2] = 0
            self.data['valid_features'][idx][no_movement_indices[0], -1] = 0
            features_idx = [0, 1, 2, 3, 4, 5]

            valid_y = self.data['valid_features'][idx][:, [-3, -2, -1]].astype(np.float32)
            for order in self.params['K_orders']:
                    for feature_idx in self.params['features']:
                        features_idx.append(6 + 4 * (order - 1) + feature_idx)
            valid_features = self.data['valid_features'][idx][:, features_idx].astype(np.float32)
            valid_x = torch.from_numpy(valid_features)
            valid_graph = Data(x=valid_x, edge_index=valid_edge_index)
            valid_graph.y = torch.from_numpy(valid_y)
            valid_set.append(valid_graph)
        data_loader['val'] = DataLoader(dataset=valid_set, batch_size=1, shuffle=False,
                                          drop_last=True)

        test_set = list()
        test_adj = self.data['test_adj_matrix']
        print(len(test_adj))
        for idx in range(len(test_adj)):
            source_array, target_array, _ = find(test_adj[idx])
            test_edge_index = torch.tensor([source_array.tolist(), target_array.tolist()], dtype=torch.long)

            no_movement_indices = np.where((self.data['test_features'][idx][:, -3] == 0) | (self.data['test_features'][idx][:, -3] == 1) )
            self.data['test_features'][idx][no_movement_indices[0], -2] = 0
            self.data['test_features'][idx][no_movement_indices[0], -1] = 0
            features_idx = [0, 1, 2, 3, 4, 5]

            test_y = self.data['test_features'][idx][:, [-3, -2, -1]].astype(np.float32)
            for order in self.params['K_orders']:
                    for feature_idx in self.params['features']:
                        features_idx.append(6 + 4 * (order - 1) + feature_idx)
            test_features = self.data['test_features'][idx][:, features_idx].astype(np.float32)
            test_x = torch.from_numpy(test_features)
            test_graph = Data(x=test_x, edge_index=test_edge_index)
            test_graph.y = torch.from_numpy(test_y)
            test_set.append(test_graph)
        data_loader['test'] = DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                          drop_last=True)
        return data_loader

    def get_model(self):
        if self.params['task'] == 'Bldgs_Gen':
            in_channels = len(self.params['K_orders']) * len(self.params['features'])
            # model = BuildingGenModel(in_channels, self.params['hidden_dims'], self.data_loader['train_deg'], self.params['dropout'])
            model = BuildingGenModel(in_channels, self.params['hidden_dims'], self.params['dropout'], self.params['model'])
        else:
            raise NotImplementedError('Invalid task.')
        if self.params['mode'] == 'test':
            model_file = self.params['output_dir'] + '/' + self.params['model_file']
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def get_cls_loss(self):
        if self.params['cls_loss'] == 'NLL':
            # [1.0, 0.11, 0.75] [1.0, 1.0, 1.0]
            class_weight = torch.FloatTensor([1.0, 1.0, 1.0])
            criterion = nn.NLLLoss(weight = class_weight, reduction='mean')
        # elif self.params['cls_loss'] == 'Focal':
        #     criterion = FocalLoss(alpha=0.9, gamma=2.0)
        # elif self.params['cls_loss'] == 'Dice':
        #     criterion = DiceLoss()
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_reg_loss(self):
        if self.params['reg_loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['reg_loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['reg_loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'],  weight_decay=self.params['weight_decay'])
            if self.params['mode'] == 'test':
                model_file = self.params['output_dir'] + '/' + self.params['model_file']
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(model_file, map_location=device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def get_deg(self, dataset):
        max_degree = -1
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg


    def train(self):
        train_history = dict()
        train_history['epoch'] = list()
        train_history['train_loss'] = list()
        train_history['val_loss'] = list()
        train_history['simplification_acc'] = list()
        # train_history['movement_acc'] = list()
        train_history['pre_distance_mse'] = list()
        train_history['next_distance_mse'] = list()
        train_history['pre_distance_mse_filter'] = list()
        train_history['next_distance_mse_filter'] = list()

        train_data = self.data_loader['train']
        val_loss = np.inf
        weights = []
        total_losses = []
        task_losses = []
        loss_ratios = []
        grad_norm_losses = []
        print("-----------Start training Building Simplification model-----------")
        for epoch in range(1, 1 + self.params['num_epochs']):

            starttime = datetime.now()
            checkpoint = {'epoch': 0, 'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'cls_loss': self.cls_criterion,
                          'reg_loss': self.reg_criterion}

            self.model.train()
            epoch_total_loss = []
            epoch_task_loss = []
            train_acc = 0.0
            batches = 0
            for batch in train_data:
                x = batch.x[:, 6:].float()
                # print(x.size())
                edge_index = batch.edge_index
                # print(edge_index.size())
                pred_rm, pred_preMove, pred_nextMove = self.model(x, edge_index)
                if self.params['cls_loss'] == 'Dice':
                    pred_rm = torch.exp(pred_rm.max(1)[0])
                rm_loss = self.cls_criterion(pred_rm, batch.y[:, 0].long())
                # preMove_loss = self.reg_criterion(pred_preMove, batch.y[:, 1])
                # nextMove_loss = self.reg_criterion(pred_nextMove, batch.y[:, 2])

                loss_indices = pred_preMove  !=  batch.y[:, 1]
                preMove_loss = self.reg_criterion(pred_preMove[(loss_indices).nonzero()], batch.y[:, 1][(loss_indices).nonzero()])
                loss_indices = pred_nextMove  !=  batch.y[:, 2]
                nextMove_loss = self.reg_criterion(pred_nextMove[(loss_indices).nonzero()], batch.y[:, 2][(loss_indices).nonzero()])

                task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss))
                total_loss = automatic_weight(self.model, task_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss.append(total_loss)
                epoch_task_loss.append(task_loss)

                batches += 1
            epoch_total_loss = torch.stack(epoch_total_loss)
            total_losses.append(torch.mean(epoch_total_loss))
            epoch_task_loss = torch.stack(epoch_task_loss)
            task_losses.append(torch.mean(epoch_task_loss))

            eval_loss, eval_removal_dict, eval_move_dict, eval_preMove_dict, eval_nextMove_dict = self.evaluation(self.data_loader['val'])

            if total_losses[-1] <= val_loss:
                print(f'Epoch {epoch}, training loss drops from {val_loss:.5} to {total_losses[-1]:.5}.  ',
                      f'Validation node removal accuracy {eval_removal_dict["accuracy"]:.5}.  ',
                    #   f'Validation node movement accuracy {eval_move_dict["accuracy"]:.5}.'
                      f'Validation rotation angle mse {eval_preMove_dict["mse"]:.5}.',
                      f'Validation move distance mse {eval_nextMove_dict["mse"]:.5}.',
                      f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s', flush=True)

                val_loss = total_losses[-1]
                checkpoint.update(epoch=epoch, model_state_dict=self.model.state_dict(), optimizer_state_dict = self.optimizer.state_dict())

                orders = ''
                for i in range(len(self.params["K_orders"])):
                    orders += str(self.params["K_orders"][i])
                    if i < len(self.params["K_orders"]) - 1:
                        orders += '&'
                torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["task"]}_{self.params["batch_size"]}_{orders}.pkl')
            else:
                print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                      f'used {(datetime.now() - starttime).seconds}s', flush=True)
            #
            train_history['epoch'].append(epoch)
            train_history['train_loss'].append(total_losses[-1].item())
            train_history['val_loss'].append(eval_loss.item())
            train_history['simplification_acc'].append(eval_removal_dict["accuracy"])
            # train_history['movement_acc'].append(eval_move_dict["accuracy"])
            train_history['pre_distance_mse'].append(eval_preMove_dict["mse"])
            train_history['next_distance_mse'].append(eval_nextMove_dict["mse"])
            train_history['pre_distance_mse_filter'].append(eval_preMove_dict["mse_filter"])
            train_history['next_distance_mse_filter'].append(eval_nextMove_dict["mse_filter"])

        pd.DataFrame(train_history).to_csv(
            self.params['output_dir'] + '/training_history_{}.csv'.format(datetime.now().strftime('%m%d%Y%H%M%S')))

        # self.test()

    def test(self):
        self.params['mode'] = 'test'
        _, eval_removal_dict, eval_move_dict, eval_preMove_dict, eval_nextMove_dict = self.evaluation(self.data_loader['test'])
        print(f'Test accuracy of simplification is: {eval_removal_dict["accuracy"]}. Test F1 of simplification is: {eval_removal_dict["f1"]}',
                # f'Test accuracy of movement is: {eval_move_dict["accuracy"]}. Test F1 of movement is: {eval_move_dict["f1"]}',
              f'Test RMSE of rotation angle is: {eval_preMove_dict["rmse"]}. Test MSE of move distance is: {eval_nextMove_dict["rmse"]}. ',
              f'Filtered test MSE of rotation angle is: {eval_preMove_dict["rmse_filter"]}. Filter test MSE of move distance is: {eval_nextMove_dict["rmse_filter"]}.', flush=True)

    def evaluation(self, dataloader):
        self.model.eval()
        if self.params['mode'] == 'test':
            self.params['dropout'] = 0.0

        rm_pred_list = list()
        rm_gt_list = list()

        mv_pred_list = list()
        mv_gt_list = list()


        eval_removal_dict = dict()
        eval_move_dict = dict()
        eval_preMove_dict = dict()
        eval_nextMove_dict = dict()

        gpd_points_list = list()

        total_losses = list()

        with torch.no_grad():
            batch_idx = 0
            # rtDirection_accuracy = []
            preMove_mse = []
            nextMove_mse = []
            preMove_mse_filter = []
            nextMove_mse_filter = []

            ds_idx = 0
            # ds_length = len(dataloader.dataset[ds_idx].indices)
            for data in dataloader:
                pred_rm, pred_preMove, pred_nextMove = self.model(data.x[:, 6:].float(), data.edge_index)
                # if self.params['cls_loss'] == 'Dice':
                #     pred_rm = torch.exp(pred_rm.max(1)[0])
                rm_pred_list.append(pred_rm)
                rm_gt_list.append(data.y[:, 0].long())
                rm_loss = self.cls_criterion(pred_rm, data.y[:, 0].long())
                pred_rm = pred_rm.max(1)[1].type_as(data.y[:, 0].long())

                preMove_loss = self.reg_criterion(pred_preMove, data.y[:, 1])
                nextMove_loss = self.reg_criterion(pred_nextMove, data.y[:, 2])

                preMove_mse.append(preMove_loss)
                nextMove_mse.append(nextMove_loss)

                task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss))
                total_loss = automatic_weight(self.model, task_loss)
                total_losses.append(total_loss)

                dis_filter = (torch.abs(data.y[:, 1]) + torch.abs(data.y[:, 2]))  >= torch.tensor(0.01)
                if dis_filter.nonzero().size(dim=0) != 0:
                     preMove_mse_filter.append(self.reg_criterion(pred_preMove[(dis_filter).nonzero()], data.y[:, 1][(dis_filter).nonzero()]))
                     nextMove_mse_filter.append(self.reg_criterion(pred_nextMove[(dis_filter).nonzero()], data.y[:, 2][(dis_filter).nonzero()]))


                if self.params['mode'] == 'test':
                    # print(data.x[0][0].item())
                    gpd_points_list.append(MTL_recontruct_points2(torch.cat([data.x, data.y], dim=1).numpy(), pred_rm, pred_preMove, pred_nextMove))

                batch_idx += 1

            eval_removal_dict['accuracy'] =  accuracy(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_pre = precision(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_rec = recall(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_removal_dict['f1'] = 2 * (eval_pre * eval_rec) / (eval_pre + eval_rec)

            preMove_mse = torch.stack(preMove_mse)
            eval_preMove_dict['mse'] = torch.mean(preMove_mse).item()
            eval_preMove_dict['rmse'] = torch.mean(torch.sqrt(preMove_mse)).item()
            nextMove_mse = torch.stack(nextMove_mse)
            eval_nextMove_dict['mse'] = torch.mean(nextMove_mse).item()
            eval_nextMove_dict['rmse'] = torch.mean(torch.sqrt(nextMove_mse)).item()

            preMove_mse_filter = torch.stack(preMove_mse_filter)
            eval_preMove_dict['mse_filter'] = torch.mean(preMove_mse_filter).item()
            eval_preMove_dict['rmse_filter'] = torch.mean(torch.sqrt(preMove_mse_filter)).item()
            nextMove_rmse_filter = torch.stack(nextMove_mse_filter)
            eval_nextMove_dict['mse_filter'] = torch.mean(nextMove_rmse_filter).item()
            eval_nextMove_dict['rmse_filter'] = torch.mean(torch.sqrt(nextMove_rmse_filter)).item()
            if len(gpd_points_list) > 0:
                gpd_points = gpd.GeoDataFrame(pd.concat(gpd_points_list, ignore_index=True))
                gpd_points.to_file(self.params['output_dir'] + '/{}_prediction.shp'.format(self.params['task']))

            eval_loss = sum(total_losses)/len(total_losses)

        return eval_loss, eval_removal_dict, eval_move_dict, eval_preMove_dict, eval_nextMove_dict