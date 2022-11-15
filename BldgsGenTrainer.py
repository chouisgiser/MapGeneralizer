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
from models import NodeClsGraphSAGE, NodeRegGraphSAGE, BuildingGenModel, BuildingVecMoveModel, BldgsGenJointModel, BldgsVecMoveJointModel, BldgsRmJointModel, BuildingGenRegModel
from scipy.sparse import find
from torch.utils.data import random_split, ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
torch.manual_seed(0)
import geopandas as gpd
from Losses import grad_norm, automatic_weight, IoULoss, HausdorffDisLoss, TurningFncLoss, FocalLoss, DiceLoss
# from focal_loss.focal_loss import FocalLoss
from torch_geometric.utils import dense_to_sparse, degree

# loss: autoweight
# tasks: node removal, vector rotation direction, vector move distance
class MTL_BuildingGen(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.cls_criterion = self.get_cls_loss()
        self.reg_criterion = self.get_reg_loss()
        self.iou_criterion = IoULoss()
        self.hdf_criterion = HausdorffDisLoss()
        self.tf_criterion = TurningFncLoss()
        self.optimizer = self.get_optimizer()

    # def get_data_loader(self):
    #     sf_data_list = list()
    #     no_sf_data_list = list()

    #     adj = self.data['adj_matrix']
    #     # for i in range(len(self.data['features'])):
    #     #     print(self.data['features'][i][0][0])
    #     Y = list()
    #     selected_indices = list()
        
    #     simplified_num = 0
    #     no_simplified_num = 0
        
    #     for idx in range(len(adj)):
    #         source_array, target_array, _ = find(adj[idx])
    #         edge_index = torch.tensor([source_array.tolist(), target_array.tolist()], dtype=torch.long)
            
    #         is_simplified = np.where((self.data['features'][idx][:, -3] == 0) | (self.data['features'][idx][:, -3] == 2) )
    #         no_simplified = np.where(self.data['features'][idx][:, -3] == 1)
    #         no_movement_indices = np.where((self.data['features'][idx][:, -3] == 0) | (self.data['features'][idx][:, -3] == 1) )
    #         self.data['features'][idx][no_movement_indices[0], -2] = 0
    #         self.data['features'][idx][no_movement_indices[0], -1] = 0
                
    #         features_idx = [0, 1, 2, 3, 4, 5]
    #         if len(is_simplified[0]) > 0:
    #             y = self.data['features'][idx][:, [-3, -2, -1]].astype(np.float32)
    #             selected_indices.append(idx)
    #             Y.append(y)
    #             for order in self.params['K_orders']:
    #                 for feature_idx in self.params['features']:
    #                     features_idx.append(6 + 4 * (order - 1) + feature_idx)
    #             features = self.data['features'][idx][:, features_idx]
    #             x = torch.from_numpy(features)
    #             graph_data = Data(x=x, edge_index=edge_index)
    #             graph_data.y = torch.from_numpy(y)
    #             sf_data_list.append(graph_data)
                
    #             simplified_num += 1
    #         # = simplified_num
    #         if len(no_simplified[0]) == len(self.data['features'][idx]) and no_simplified_num < 0:
    #             y = self.data['features'][idx][:, [-3, -2, -1]].astype(np.float32)
    #             selected_indices.append(idx)
    #             Y.append(y)
    #             for order in self.params['K_orders']:
    #                 for feature_idx in self.params['features']:
    #                     features_idx.append(6 + 4 * (order - 1) + feature_idx)
    #             features = self.data['features'][idx][:, features_idx]
    #             x = torch.from_numpy(features).float()
    #             graph_data = Data(x=x, edge_index=edge_index)
    #             graph_data.y = torch.from_numpy(y)
    #             no_sf_data_list.append(graph_data)
                
    #             no_simplified_num += 1

    #     self.data['features'] = self.data['features'][selected_indices]
    #     self.data['adj_matrix'] = self.data['adj_matrix'][selected_indices]
    #     data_loader = dict()
    #     dataset_size = len(no_sf_data_list) + len(sf_data_list)
    #     print(dataset_size)
    #     Y = np.concatenate(Y, axis=0)
    #     print(Y.shape[0])
    #     print(Y[:, 0].sum())
    #     train_size = int(self.params['split_ratio'][0] / sum(self.params['split_ratio']) * len(sf_data_list))
    #     val_size = int(self.params['split_ratio'][1] / sum(self.params['split_ratio']) * len(sf_data_list))
    #     test_size = len(sf_data_list) - train_size - val_size
    #     sf_train_set, sf_val_set, sf_test_set = random_split(sf_data_list, [train_size, val_size, test_size],
    #                                       generator=torch.Generator().manual_seed(0))
        
    #     train_size = int(self.params['split_ratio'][0] / sum(self.params['split_ratio']) * len(no_sf_data_list))
    #     val_size = int(self.params['split_ratio'][1] / sum(self.params['split_ratio']) * len(no_sf_data_list))
    #     test_size = len(no_sf_data_list) - train_size - val_size
    #     no_sf_train_set, no_sf_val_set, no_sf_test_set = random_split(no_sf_data_list, [train_size, val_size, test_size],
    #                                       generator=torch.Generator().manual_seed(0))
    #     # data_loader['train_deg'] = self.get_deg(dataset=ConcatDataset([sf_train_set, no_sf_train_set]))
    #     data_loader['train'] = DataLoader(dataset=ConcatDataset([sf_train_set, no_sf_train_set]), batch_size=self.params['batch_size'], shuffle=True,
    #                                       drop_last=True)
    #     data_loader['val'] = DataLoader(dataset=ConcatDataset([sf_val_set, no_sf_val_set]), batch_size=1, shuffle=False,
    #                                       drop_last=True)
    #     data_loader['test'] = DataLoader(dataset=ConcatDataset([sf_test_set, no_sf_test_set]), batch_size=1, shuffle=False,
    #                                      drop_last=True)

    #     return data_loader
    
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
        elif self.params['cls_loss'] == 'Focal':
            criterion = FocalLoss(alpha=0.9, gamma=2.0)
        elif self.params['cls_loss'] == 'Dice':
            criterion = DiceLoss()
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
        print("-----------Start training GCAE model-----------")
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
                
                # pred_labels = pred_rm.max(1)[1].type_as(batch.y[:, 0].long())
                # gt_areas, pred_areas, gt_inangle_sums, pred_inangle_sums = MTL_reconstruct_polygon(batch.x[:, 0:4], pred_labels, pred_preMove, pred_nextMove, batch.y)
                
                # area_loss = self.reg_criterion(pred_areas, gt_areas)
                # task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss, area_loss))
                # total_loss = automatic_weight(self.model, task_loss)
                
                # inangle_sum_loss = self.reg_criterion(pred_inangle_sums, gt_inangle_sums)
                # task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss, inangle_sum_loss))
                # total_loss = automatic_weight(self.model, task_loss)
                
                # pred_labels = pred_rm.max(1)[1].type_as(batch.y[:, 0].long())
                # iou_loss = self.iou_criterion(batch.x[:, 0:4], pred_labels, pred_preMove, pred_nextMove, batch.y)
                # task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss, iou_loss))
                # total_loss = automatic_weight(self.model, task_loss)

                
                # pred_labels = pred_rm.max(1)[1].type_as(batch.y[:, 0].long())
                # hdf_loss = self.hdf_criterion(batch.x[:, 0:4], pred_labels, pred_preMove, pred_nextMove, batch.y)
                # task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss, hdf_loss))
                # total_loss = automatic_weight(self.model, task_loss)
                
                # pred_labels = pred_rm.max(1)[1].type_as(batch.y[:, 0].long())
                # tf_loss = self.tf_criterion(batch.x[:, 0:4], pred_labels, pred_preMove, pred_nextMove, batch.y)
                # task_loss = torch.stack((rm_loss, preMove_loss, nextMove_loss, tf_loss))
                # total_loss = automatic_weight(self.model, task_loss)
                
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
            
            # eval_move_dict['accuracy'] =  accuracy(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_pre_mv = precision(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_rec_mv = recall(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_move_dict['f1'] = 2 * (eval_pre_mv * eval_rec_mv) / (eval_pre_mv+ eval_rec_mv)
            
            # rtDirection_loss = torch.stack(rtDirection_accuracy)
            # eval_removal_dict['accuracy'] = torch.mean(rtDirection_loss).item()
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

class MTL_BuildingGenReg(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.reg_criterion = self.get_reg_loss()
        self.iou_criterion = IoULoss()
        self.hdf_criterion = HausdorffDisLoss()
        self.tf_criterion = TurningFncLoss()
        self.optimizer = self.get_optimizer()

    def get_data_loader(self):
        data_loader = dict()
        
        train_set = list()
        train_adj = self.data['train_adj_matrix']
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
            model = BuildingGenRegModel(in_channels, self.params['hidden_dims'], self.params['dropout'])
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

    def train(self):
        train_history = dict()
        train_history['epoch'] = list()
        train_history['train_loss'] = list()
        # train_history['simplification_acc'] = list()
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
        print("-----------Start training GCAE model-----------")
        for epoch in range(1, 1 + self.params['num_epochs']):

            starttime = datetime.now()
            checkpoint = {'epoch': 0, 'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'reg_loss': self.reg_criterion}

            self.model.train()
            epoch_total_loss = []
            epoch_task_loss = []
            train_acc = 0.0
            batches = 0
            for batch in train_data:
                x = batch.x[:, 6:]
                edge_index = batch.edge_index
                pred_preMove, pred_nextMove = self.model(x, edge_index)
                preMove_loss = self.reg_criterion(pred_preMove, batch.y[:, 1])
                nextMove_loss = self.reg_criterion(pred_nextMove, batch.y[:, 2])
                
                task_loss = torch.stack((preMove_loss, nextMove_loss))
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

            eval_removal_dict, eval_preMove_dict, eval_nextMove_dict = self.evaluation(self.data_loader['val'])

            if total_losses[-1] <= val_loss:
                print(f'Epoch {epoch}, training loss drops from {val_loss:.5} to {total_losses[-1]:.5}.  ',
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
            train_history['train_loss'].append(total_losses[-1])
            # train_history['simplification_acc'].append(eval_removal_dict["accuracy"])
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
        eval_removal_dict, eval_preMove_dict, eval_nextMove_dict = self.evaluation(self.data_loader['test'])
        print(f'Test RMSE of rotation angle is: {eval_preMove_dict["rmse"]}. Test MSE of move distance is: {eval_nextMove_dict["rmse"]}. ',
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

        with torch.no_grad():
            batch_idx = 0
            # rtDirection_accuracy = []
            preMove_mse = []
            nextMove_mse = []
            preMove_mse_filter = []
            nextMove_mse_filter = []
            
            ds_idx = 0
            # ds_length = len(dataloader.dataset.datasets[ds_idx].indices)
            for data in dataloader:
                pred_preMove, pred_nextMove = self.model(data.x[:, 6:], data.edge_index)
                # rtDirection_accuracy.append(accuracy(pred_rtDirection, data.y[:, 0].long()))
                # rm_pred_list.append(pred_rm)
                # rm_gt_list.append(data.y[:, 0].long())
                # pred_rm = pred_rm.max(1)[1].type_as(data.y[:, 0].long())
                
                # mv_pred_list.append(pred_mv)
                # mv_gt_list.append(data.y[:, 0].long())
                # pred_mv = pred_mv.max(1)[1].type_as(data.y[:, 1].long())

                preMove_mse.append(self.reg_criterion(pred_preMove, data.y[:, 1]))
                nextMove_mse.append(self.reg_criterion(pred_nextMove, data.y[:, 2]))

                dis_filter = (torch.abs(data.y[:, 1]) + torch.abs(data.y[:, 2]))  >= torch.tensor(0.01)
                if dis_filter.nonzero().size(dim=0) != 0:
                     preMove_mse_filter.append(self.reg_criterion(pred_preMove[(dis_filter).nonzero()], data.y[:, 1][(dis_filter).nonzero()]))
                     nextMove_mse_filter.append(self.reg_criterion(pred_nextMove[(dis_filter).nonzero()], data.y[:, 2][(dis_filter).nonzero()]))
                    
                # pred_rtAngle_filter = pred_rtAngle[(dis_filter).nonzero()]
                # gt_rtAngle_filter = data.y[:, 1][(dis_filter).nonzero()]
                # if gt_rtAngle_filter.size()[0] != 0 and gt_rtAngle_filter.size()[1] != 0:
                #     rtAngle_mse_filter.append(self.reg_criterion(pred_rtAngle_filter, gt_rtAngle_filter))

                # pred_moveDis_filter = pred_moveDis[(dis_filter).nonzero()]
                # gt_moveDis_filter = data.y[:, 2][(dis_filter).nonzero()]
                # if gt_moveDis_filter.size()[0] != 0 and gt_moveDis_filter.size()[1] != 0:
                #     moveDis_mse_filter.append(self.reg_criterion(pred_moveDis_filter, gt_moveDis_filter))
                if self.params['mode'] == 'test':
                    gpd_points_list.append(MTL_recontruct_points3(torch.cat([data.x, data.y], dim=1).numpy(), pred_preMove, pred_nextMove))

                batch_idx += 1
            
            eval_removal_dict['accuracy'] =  0.0
            # eval_pre = precision(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            # eval_rec = recall(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_removal_dict['f1'] = 0.0
            
            # eval_move_dict['accuracy'] =  accuracy(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_pre_mv = precision(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_rec_mv = recall(torch.cat(mv_pred_list), torch.cat(mv_gt_list)).item()
            # eval_move_dict['f1'] = 2 * (eval_pre_mv * eval_rec_mv) / (eval_pre_mv+ eval_rec_mv)
            
            # rtDirection_loss = torch.stack(rtDirection_accuracy)
            # eval_removal_dict['accuracy'] = torch.mean(rtDirection_loss).item()
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

        return eval_removal_dict, eval_preMove_dict, eval_nextMove_dict

class MTL_BuildingRmJoint(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.cls_criterion = self.get_cls_loss()
        self.reg_criterion = self.get_reg_loss()
        self.optimizer = self.get_optimizer()

    def get_data_loader(self):
        data_list = list()

        adj = self.data['adj_matrix']
        Y = list()
        selected_indices = list()
        for idx in range(len(adj)):
            source_array, target_array, _ = find(adj[idx])
            edge_index = torch.tensor([source_array.tolist(), target_array.tolist()], dtype=torch.long)
            # self.data['features'][idx][:, -2] = self.data['features'][idx][:, -2] * math.pi / 180
            is_simplified = np.where((np.absolute(self.data['features'][idx][:, -2]) + np.absolute(self.data['features'][idx][:, -1])) > 0.0)
            if len(is_simplified[0]) > 0:
                y = self.data['features'][idx][:, [-3, -2, -1]].astype(np.float32)
                selected_indices.append(idx)
                Y.append(y)
                features_idx = [0, 1, 2, 3]
                for order in self.params['K_orders']:
                    for feature_idx in self.params['features']:
                        features_idx.append(4 + len(self.params['features']) * (order - 1) + (feature_idx - 1))
                features = self.data['features'][idx][:, features_idx]
                x = torch.from_numpy(features.astype(np.double)).float()
                graph_data = Data(x=x, edge_index=edge_index)
                graph_data.y = torch.from_numpy(y)
                data_list.append(graph_data)

        self.data['features'] = self.data['features'][selected_indices]
        self.data['adj_matrix'] = self.data['adj_matrix'][selected_indices]
        data_loader = dict()
        dataset_size = len(data_list)
        print(dataset_size)
        Y = np.concatenate(Y, axis=0)
        print(Y.shape[0])
        print(Y[:, 0].sum())
        train_size = int(self.params['split_ratio'][0] / sum(self.params['split_ratio']) * dataset_size)
        val_size = int(self.params['split_ratio'][1] / sum(self.params['split_ratio']) * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_set, val_set, test_set = random_split(data_list, [train_size, val_size, test_size],
                                           generator=torch.Generator().manual_seed(0))
        data_loader['train'] = DataLoader(dataset=train_set, batch_size=self.params['batch_size'], shuffle=True,
                                          drop_last=True)
        data_loader['val'] = DataLoader(dataset=val_set, batch_size=1, shuffle=False,
                                          drop_last=True)
        data_loader['test'] = DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                         drop_last=True)
        return data_loader

    def get_model(self):
        if self.params['task'] == 'Bldgs_Gen':
            in_channels = len(self.params['K_orders']) * len(self.params['features'])
            model = BldgsRmJointModel(in_channels, self.params['hidden_dims'], self.params['dropout'], self.params['model'])
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
            class_weight = torch.FloatTensor([1.0, 1.0, 1.0]) 
            criterion = nn.NLLLoss(weight = class_weight, reduction='mean')
        elif self.params['cls_loss'] == 'Focal':
            criterion = FocalLoss(alpha=0.1, gamma=2)
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

    def train(self):
        train_history = dict()
        train_history['epoch'] = list()
        train_history['train_loss'] = list()
        train_history['simplification_acc'] = list()
        train_history['jointRt_mse'] = list()
        train_history['jointRt_mse_filter'] = list()
        train_history['jointMove_mse'] = list()
        train_history['jointMove_mse_filter'] = list()

        train_data = self.data_loader['train']
        val_loss = np.inf
        weights = []
        total_losses = []
        task_losses = []
        loss_ratios = []
        grad_norm_losses = []
        print("-----------Start training GCAE model-----------")
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
                x = batch.x[:, 6:]
                edge_index = batch.edge_index
                pred_rm, pred_jointRtMove = self.model(x, edge_index)
                
                if self.params['cls_loss'] == 'Focal':
                    pred_rm = pred_rm.max(1)[0]
                    
                rm_loss = self.cls_criterion(pred_rm, batch.y[:, 0].long())
                jointRtMove_loss = self.reg_criterion(pred_jointRtMove, torch.flatten(batch.y[:, [1, 2]]))
                task_loss = torch.stack((rm_loss, jointRtMove_loss))
                total_loss = automatic_weight(self.model, task_loss)
                # total_loss = rtDirection_loss

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

            eval_removal_dict, eval_jointRtMove_dict = self.evaluation(self.data_loader['val'])

            if total_losses[-1] <= val_loss:
                print(f'Epoch {epoch}, training loss drops from {val_loss:.5} to {total_losses[-1]:.5}.  '
                      f'Validation node removal accuracy {eval_removal_dict["accuracy"]:.5}.',
                      f'Validation joint rotation mse {eval_jointRtMove_dict["rt_mse"]:.5}. ',
                      f'Validation joint move mse {eval_jointRtMove_dict["move_mse"]:.5}. ',
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
            train_history['train_loss'].append(total_losses[-1])
            train_history['simplification_acc'].append(eval_removal_dict["accuracy"])
            train_history['jointRt_mse'].append(eval_jointRtMove_dict["rt_mse"])
            train_history['jointRt_mse_filter'].append(eval_jointRtMove_dict["rt_mse_filter"])
            train_history['jointMove_mse'].append(eval_jointRtMove_dict["move_mse"])
            train_history['jointMove_mse_filter'].append(eval_jointRtMove_dict["move_mse_filter"])

        pd.DataFrame(train_history).to_csv(
            self.params['output_dir'] + '/training_history_{}.csv'.format(datetime.now().strftime('%m%d%Y%H%M%S')))

        # self.test()

    def test(self):
        self.params['mode'] = 'test'
        eval_removal_dict, eval_jointRtMove_dict = self.evaluation(self.data_loader['test'])
        print(f'Test accuracy of simplification is: {eval_removal_dict["accuracy"]}. Test F1 of simplification is: {eval_removal_dict["f1"]}',
              f'Test RMSE of joint rotation angle is: {eval_jointRtMove_dict["rt_rmse"]}. Test RMSE of joint move distance is: {eval_jointRtMove_dict["move_rmse"]}. ',
              f'Filtered test RMSE of joint rotation angle is: {eval_jointRtMove_dict["rt_rmse_filter"]}. Filter test RMSE of joint move distance is: {eval_jointRtMove_dict["move_rmse_filter"]}.', flush=True)

    def evaluation(self, dataloader):
        self.model.eval()
        eval_removal_dict = dict()
        eval_jointRtMove_dict = dict()

        gpd_points_list = list()
        
        with torch.no_grad():
            batch_idx = 0
            rm_pred_list = list()
            rm_gt_list = list()
            jointRt_mse = []
            jointMove_mse = []
            jointRt_mse_filter = []
            jointMove_mse_filter = []
            for data in dataloader:
                if self.params['mode'] == 'test':
                    self.params['dropout'] = 0.0
                    
                _, pred_rm, pred_jointRtMove = self.model(data.x[:, 6:], data.edge_index)
                
                rm_pred_list.append(pred_rm)
                rm_gt_list.append(data.y[:, 0].long())
                # rm_accuracy.append(accuracy(pred_rm, data.y[:, 0].long()))
                pred_rm = pred_rm.max(1)[1].type_as(data.y[:, 0].long())
                
                rt_idx = [2 * i for i in range(int(pred_jointRtMove.size(dim=0) / 2))]
                move_idx = [2 * i + 1 for i in range(int(pred_jointRtMove.size(dim=0) / 2))]
                jointRt_mse.append(self.reg_criterion(pred_jointRtMove[rt_idx], data.y[:, 1]))
                jointMove_mse.append(self.reg_criterion(pred_jointRtMove[move_idx], data.y[:, 2]))

                dis_filter = torch.abs(data.y[:, 2]) >= torch.tensor(0.01)
                if dis_filter.nonzero().size(dim=0) != 0:

                    rt_idx_filter = dis_filter.nonzero() * 2
                    move_idx_filter = dis_filter.nonzero() * 2 + 1
                    jointRt_mse_filter.append(self.reg_criterion(pred_jointRtMove[rt_idx_filter],
                                                                 torch.flatten(data.y[:, [1, 2]])[rt_idx_filter]))
                    jointMove_mse_filter.append(self.reg_criterion(pred_jointRtMove[move_idx_filter],
                                                                   torch.flatten(data.y[:, [1, 2]])[move_idx_filter]))
                
                if self.params['mode'] == 'test':
                    idx = dataloader.dataset.indices[batch_idx]
                    graph_features = self.data['features'][idx]
                    gpd_points_list.append(MTL_recontruct_points2(graph_features, pred_rm, pred_jointRtMove[rt_idx], pred_jointRtMove[move_idx]))

                batch_idx += 1

            # rm_loss = torch.stack(rm_accuracy)
            eval_removal_dict['accuracy'] =  accuracy(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_pre = precision(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_rec = recall(torch.cat(rm_pred_list), torch.cat(rm_gt_list)).item()
            eval_removal_dict['f1'] = 2 * (eval_pre * eval_rec) / (eval_pre + eval_rec)

            jointRt_mse = torch.stack(jointRt_mse)
            eval_jointRtMove_dict['rt_mse'] = torch.mean(jointRt_mse).item()
            eval_jointRtMove_dict['rt_rmse'] = torch.mean(torch.sqrt(jointRt_mse)).item()
            jointMove_mse = torch.stack(jointMove_mse)
            eval_jointRtMove_dict['move_mse'] = torch.mean(jointMove_mse).item()
            eval_jointRtMove_dict['move_rmse'] = torch.mean(torch.sqrt(jointMove_mse)).item()

            jointRt_mse_filter = torch.stack(jointRt_mse_filter)
            eval_jointRtMove_dict['rt_mse_filter'] = torch.mean(jointRt_mse_filter).item()
            eval_jointRtMove_dict['rt_rmse_filter'] = torch.mean(torch.sqrt(jointRt_mse_filter)).item()
            jointMove_mse_filter = torch.stack(jointMove_mse_filter)
            eval_jointRtMove_dict['move_mse_filter'] = torch.mean(jointMove_mse_filter).item()
            eval_jointRtMove_dict['move_rmse_filter'] = torch.mean(torch.sqrt(jointMove_mse_filter)).item()
            if len(gpd_points_list) > 0:
                gpd_points = gpd.GeoDataFrame(pd.concat(gpd_points_list, ignore_index=True))
                gpd_points.to_file(self.params['output_dir'] + '/{}_prediction.shp'.format(self.params['task']))

        return eval_removal_dict, eval_jointRtMove_dict

class SFModelTrainer(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.criterion = self.get_cls_loss()
        self.optimizer = self.get_optimizer()

    def get_data_loader(self):
        data_loader = dict()
        
        train_set = list()
        train_adj = self.data['train_adj_matrix']
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
        if self.params['task'] == 'Shape_Encoding':
            in_channels, out_channels = len(self.params['K_order']) * 9, len(self.params['K_order']) * 9
            encoder = GCAEncoder(in_channels, self.params['hidden_layers'])
            decoder = GCADecoder(self.params['hidden_layers'], out_channels)
            model = GAE(encoder=encoder, decoder=decoder)
        elif self.params['task'] == 'Node_removal':
            in_channels, out_channels = len(self.params['K_orders']) * len(self.params['features']), 3
            if self.params['model'] == 'GCN':
                model = NodeClsGCN(in_channels, self.params['hidden_dims'], out_channels, self.params['dropout'])
            elif self.params['model'] == 'GAT':
                model = NodeClsGAT(in_channels, self.params['hidden_dims'], out_channels, self.params['dropout'])
            elif self.params['model'] == 'GraphSAGE':
                model = NodeClsGraphSAGE(in_channels, self.params['hidden_dims'], out_channels, self.params['dropout'], self.params['task'])
            else:
                raise NotImplementedError('Invalid model name.')
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
            criterion = nn.NLLLoss(reduction='mean')
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

    def train(self):
        train_history = dict()
        train_history['epoch'] = list()
        train_history['train_loss'] = list()
        train_history['train_acc'] = list()
        train_history['val_loss'] = list()
        train_history['val_acc'] = list()
        train_history['val_pre'] = list()
        train_history['val_rec'] = list()
        train_history['val_F1'] = list()

        train_data = self.data_loader['train']
        val_loss = np.inf
        print("-----------Start training GCAE model-----------")
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            checkpoint = {'epoch': 0, 'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.criterion}

            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            batches = 0
            for batch in train_data:
                batches += 1
                x = batch.x[:, 6:].float()
                edge_index = batch.edge_index
                output = self.model(x, edge_index)
                batch_train_loss = self.criterion(output, batch.y[:, 0].long())
                train_loss += batch_train_loss.item()
                # train_acc += accuracy(output, batch.y).item()
                train_acc += 0
                self.optimizer.zero_grad()
                batch_train_loss.backward()
                self.optimizer.step()

            train_acc = train_acc/batches
            epoch_val_loss, val_acc, val_pre, val_rec, val_F1 = self.evaluation(self.data_loader['val'])

            if epoch_val_loss <= val_loss:
                print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                      f'Training loss {train_loss:.5}. Validation accuracy {val_acc:.5}.'
                      f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s', flush=True)

                val_loss = epoch_val_loss
                checkpoint.update(epoch=epoch, model_state_dict=self.model.state_dict(), optimizer_state_dict = self.optimizer.state_dict())

                orders = ''
                for i in range(len(self.params["K_orders"])):
                    orders += str(self.params["K_orders"][i])
                    if i < len(self.params["K_orders"]) - 1:
                        orders += '&'
                torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["task"]}_{self.params["batch_size"]}_{orders}.pkl')
            else:
                print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                      f'used {(datetime.now() - starttime).seconds}s', flush = True)

            train_history['epoch'].append(epoch)
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)
            train_history['val_pre'].append(val_pre)
            train_history['val_rec'].append(val_rec)
            train_history['val_F1'].append(val_F1)

        pd.DataFrame(train_history).to_csv(
            self.params['output_dir'] + '/training_history_{}.csv'.format(datetime.now().strftime('%m%d%Y%H%M%S')))

    def test(self):
        _, test_acc, test_pre, test_rec, test_F1 = self.evaluation(self.data_loader['test'])
        print(f'Test accuracy is: {test_acc}. Test precision is: {test_pre}. Test recall is: {test_rec}. Test F1 is: {test_F1}.', flush = True)

    def evaluation(self, dataloader):
        self.model.eval()
        output_list = list()
        y_list = list()
        eval_loss = 0.0
        gpd_points_list = list()

        with torch.no_grad():
            batch_idx = 0
            for data in dataloader:
                output = self.model(data.x[:, 6:].float(), data.edge_index)
                output_list.append(output)
                y_list.append(data.y[:, 0].long())
                eval_loss += self.criterion(output, data.y[:, 0].long()).item()
                if self.params['mode'] == 'test':
                    idx = dataloader.dataset.indices[batch_idx]
                    graph_features = self.data['features'][idx]
                    gpd_points_list.append(STL_recontruct_points(graph_features, output))
                batch_idx += 1

            # if len(gpd_points_list) > 0:
            #     gpd_points = gpd.GeoDataFrame(pd.concat(gpd_points_list, ignore_index=True))
            #     gpd_points.to_file(self.params['output_dir'] + '/{}_prediction.shp'.format(self.params['task']))

        eval_acc = accuracy(torch.cat(output_list), torch.cat(y_list)).item()
        eval_pre = precision(torch.cat(output_list), torch.cat(y_list)).item()
        eval_rec = recall(torch.cat(output_list), torch.cat(y_list)).item()
        F1 = 2 * (eval_pre * eval_rec) / (eval_pre + eval_rec)

        return eval_loss, eval_acc, eval_pre, eval_rec, F1
        
class DPModelTrainer(object):
    def __init__(self, params:dict, data_container):
        self.params = params
        self.data = data_container.load_data()
        self.data_loader = self.get_data_loader()
        self.model = self.get_model()
        self.reg_criterion = self.get_reg_loss()
        self.cls_criterion = self.get_cls_loss()
        self.optimizer = self.get_optimizer()

    def get_data_loader(self):
        data_loader = dict()
        
        train_set = list()
        train_adj = self.data['train_adj_matrix']
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
        if self.params['task'] == 'Shape_Encoding':
            in_channels, out_channels = len(self.params['K_order']) * 9, len(self.params['K_order']) * 9
            encoder = GCAEncoder(in_channels, self.params['hidden_layers'])
            decoder = GCADecoder(self.params['hidden_layers'], out_channels)
            model = GAE(encoder=encoder, decoder=decoder)
        elif self.params['task'] == 'Vec_dis':
            in_channels, out_channels = len(self.params['K_orders']) * len(self.params['features']), 1
            model = NodeRegGraphSAGE(in_channels, self.params['hidden_dims'], out_channels, self.params['task'])
        elif self.params['task'] == 'Vec_dir':
            in_channels, out_channels = len(self.params['K_orders']) * len(self.params['features']), 1
            model = NodeRegGraphSAGE(in_channels, self.params['hidden_dims'], out_channels, self.params['task'])
        elif self.params['task'] == 'Vec_dir_dis':
            in_channels, out_channels = len(self.params['K_orders']) * len(self.params['features']), 2
            model = NodeRegGraphSAGE(in_channels, self.params['hidden_dims'], out_channels, self.params['task'])
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
            criterion = nn.NLLLoss(reduction='mean')
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

    def train(self):
        train_history = dict()
        train_history['epoch'] = list()
        train_history['train_loss'] = list()
        train_history['train_acc'] = list()
        train_history['val_loss'] = list()
        # train_history['val_acc'] = list()
        # train_history['val_pre'] = list()
        # train_history['val_rec'] = list()
        # train_history['val_F1'] = list()

        train_data = self.data_loader['train']
        val_loss = np.inf
        print("-----------Start training GCAE model-----------")
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            checkpoint = {'epoch': 0, 'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.reg_criterion}

            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            batches = 0
            for batch in train_data:
                batches += 1
                x = batch.x[:, 6:].float()
                edge_index = batch.edge_index
                output = self.model(x, edge_index)
                if self.params['task'] == 'Vec_dir':
                    gt = batch.y[:, 1]
                elif self.params['task'] == 'Vec_dis':
                    gt = batch.y[:, 2]
                elif self.params['task'] == 'Vec_dir_dis':
                    gt = torch.flatten(batch.y[:, [1, 2]])
                else:
                    raise NotImplementedError('Invalid task.')
                batch_train_loss = self.reg_criterion(output, gt)
                train_loss += batch_train_loss.item()
                # train_acc += accuracy(output, batch.y).item()
                train_acc += 0
                self.optimizer.zero_grad()
                batch_train_loss.backward()
                self.optimizer.step()

            train_acc = train_acc/batches
            epoch_val_loss = self.evaluation(self.data_loader['val'])

            if epoch_val_loss['mse'] <= val_loss:
                print(f'Epoch {epoch}, training loss drops from {val_loss:.5} to {train_loss/batches:.5}.  '
                      f'Validation mse {epoch_val_loss["mse"]:.5}.'
                      f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s', flush = True)

                val_loss = epoch_val_loss['mse']

                checkpoint.update(epoch=epoch, model_state_dict=self.model.state_dict(), optimizer_state_dict = self.optimizer.state_dict())

                orders = ''
                for i in range(len(self.params["K_orders"])):
                    orders += str(self.params["K_orders"][i])
                    if i < len(self.params["K_orders"]) - 1:
                        orders += '&'
                torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["task"]}_{self.params["batch_size"]}_{orders}.pkl')
            else:
                print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                      f'used {(datetime.now() - starttime).seconds}s')

            train_history['epoch'].append(epoch)
            train_history['train_loss'].append(train_loss)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)

        pd.DataFrame(train_history).to_csv(
            self.params['output_dir'] + '/training_history_{}.csv'.format(datetime.now().strftime('%m%d%Y%H%M%S')))

    def test(self):
        eval_dp = self.evaluation(self.data_loader['test'])
        if self.params['task'] == 'Vec_dir_dis':
            print(
                f'Test RMSE of rotation angle is: {eval_dp["rt_rmse"]}. Test RMSE of move distance is: {eval_dp["move_rmse"]}. '
                f'Filtered test RMSE of rotation angle is: {eval_dp["rt_rmse_filter"]}. Filter test RMSE of move distance is: {eval_dp["move_rmse_filter"]}.', flush = True)
        print(f'Test RMSE is: {eval_dp["rmse"]}. Filter test RMSE is: {eval_dp["rmse_filter"]}.', flush = True)

    def evaluation(self, dataloader):
        self.model.eval()
        output_list = list()
        # y_list = list()
        eval_loss = 0.0
        gpd_points_list = list()
        eval_dp = dict()
        # rmse_filter = []

        with torch.no_grad():
            mse = []
            mse_filter = []
            batch_idx = 0
            rt_mse = []
            move_mse = []
            rt_mse_filter = []
            move_mse_filter = []
            for data in dataloader:
                batch_idx += 1
                output = self.model(data.x[:, 6:].float(), data.edge_index)
                # output_list.append(output)
                if self.params['task'] == 'Vec_dir':
                    gt = data.y[:, 1]
                elif self.params['task'] == 'Vec_dis':
                    gt = data.y[:, 2]
                elif self.params['task'] == 'Vec_dir_dis':
                    gt = torch.flatten(data.y[:, [1, 2]])
                else:
                    raise NotImplementedError('Invalid task.')

                mse.append(self.reg_criterion(output, gt))
                # dis_filter = torch.abs(data.y[:, 2]) >= torch.tensor(0.01)
                dis_filter = (torch.abs(data.y[:, 1]) + torch.abs(data.y[:, 2]))  >= torch.tensor(0.01)

                if self.params['task'] == 'Vec_dir_dis':
                    rt_idx = [2 * i for i in range(int(output.size(dim=0) / 2))]
                    move_idx = [2 * i + 1 for i in range(int(output.size(dim=0) / 2))]
                    rt_mse.append(
                        self.reg_criterion(output[rt_idx], gt[rt_idx]))
                    move_mse.append(self.reg_criterion(output[move_idx], gt[move_idx]))
                    if dis_filter.nonzero().size(dim=0) != 0:
                        rt_idx_filter = dis_filter.nonzero() * 2
                        move_idx_filter = dis_filter.nonzero() * 2 + 1
                        pred_filter = output[torch.stack([rt_idx_filter, move_idx_filter])]
                        gt_filter = gt[torch.stack([rt_idx_filter, move_idx_filter])]
                        mse_filter.append(self.reg_criterion(pred_filter, gt_filter))

                        rt_mse_filter.append(self.reg_criterion(output[rt_idx_filter], gt[rt_idx_filter]))
                        move_mse_filter.append(self.reg_criterion(output[move_idx_filter],gt[move_idx_filter]))
                else:
                    pred_filter = output[dis_filter.nonzero()]
                    gt_filter = gt[dis_filter.nonzero()]
                    if gt_filter.size()[0] != 0 and gt_filter.size()[1] != 0:
                        mse_filter.append(self.reg_criterion(pred_filter, gt_filter))

            mse = torch.stack(mse)
            eval_dp['mse'] = torch.mean(mse).item()
            eval_dp['rmse'] = torch.mean(torch.sqrt(mse)).item()
            mse_filter = torch.stack(mse_filter)
            eval_dp['mse_filter'] = torch.mean(mse_filter).item()
            eval_dp['rmse_filter'] = torch.mean(torch.sqrt(mse_filter)).item()
            if self.params['task'] == 'Vec_dir_dis':
                rt_mse = torch.stack(rt_mse)
                eval_dp['rt_mse'] = torch.mean(rt_mse).item()
                eval_dp['rt_rmse'] = torch.mean(torch.sqrt(rt_mse)).item()
                rt_mse_filter = torch.stack(rt_mse_filter)
                eval_dp['rt_mse_filter'] = torch.mean(rt_mse_filter).item()
                eval_dp['rt_rmse_filter'] = torch.mean(torch.sqrt(rt_mse_filter)).item()

                move_mse = torch.stack(move_mse)
                eval_dp['move_mse'] = torch.mean(move_mse).item()
                eval_dp['move_rmse'] = torch.mean(torch.sqrt(move_mse)).item()
                move_mse_filter = torch.stack(move_mse_filter)
                eval_dp['move_mse_filter'] = torch.mean(move_mse_filter).item()
                eval_dp['move_rmse_filter'] = torch.mean(torch.sqrt(move_mse_filter)).item()
                if self.params['mode'] == 'test':
                    idx = dataloader.dataset.indices[batch_idx]
                    graph_features = self.data['features'][idx]
                    gpd_points_list.append(STL_recontruct_points(graph_features, output))

            if len(gpd_points_list) > 0:
                gpd_points = gpd.GeoDataFrame(pd.concat(gpd_points_list, ignore_index=True))
                gpd_points.to_file(self.params['output_dir'] + '/{}_prediction.shp'.format(self.params['task']))

        return eval_dp
