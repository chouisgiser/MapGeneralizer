# -*- coding: utf-8 -*-
"""
# @time    : 05.05.22 15:29
# @author  : zhouzy
# @file    : main.py
"""
import argparse
from BldgsGenTrainer import MTL_BuildingGen
from utils import DataInput

map_generalizer = argparse.ArgumentParser(description='Run Polygon Graph with GAE')
map_generalizer.add_argument('-in', '--input_dir', type=str, default='data/input')
map_generalizer.add_argument('-out', '--output_dir', type=str, default='data/output')
map_generalizer.add_argument('-scale', '--src_tar', type=int, nargs='+', default= [5, 10])
map_generalizer.add_argument('-batch', '--batch_size', type=int, default=64)
map_generalizer.add_argument('-split', '--split_ratio', type=float, nargs='+',
                         help='Relative data split ratio in train : validate : test'
                              ' Example: -split 5 1 2', default=[6, 2, 2])
#   1, 4
map_generalizer.add_argument('-order', '--K_orders', type=int, nargs='+', default=[1])

map_generalizer.add_argument('-features', '--features', type=int, help='Specify feature index', default=[0, 1, 2, 3])
map_generalizer.add_argument('-hidden', '--hidden_dims', type=int, nargs='+', default=[32, 16, 8, 4])
map_generalizer.add_argument('-lr', '--learn_rate', type=float, default=5e-4)
map_generalizer.add_argument('-epoch', '--num_epochs', type=int, default=500)
map_generalizer.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cpu')
map_generalizer.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
map_generalizer.add_argument('-weight', '--weight_decay', type=float, help='Weight decay for the optimizer', default=1e-6)
map_generalizer.add_argument('-dropout', '--dropout', type=float, help='Dropout for the model', default=0.0)

map_generalizer.add_argument('-task', '--task', type=str, help='Specify task', choices=['Bldgs_Gen', 'Node_removal', 'Vec_dir', 'Vec_dis'],
                             default='Bldgs_Gen')
map_generalizer.add_argument('-model', '--model', type=str, help='Specify model', choices=['GAE', 'GCN', 'GAT', 'GraphSAGE'],
                             default='GraphSAGE')
map_generalizer.add_argument('-mode', '--mode', type=str, help='Specify stage of deep learning', choices=['train', 'test'],
                             default='train')
map_generalizer.add_argument('-cls_loss', '--cls_loss', type=str, help='Specify loss function', choices =['NLL', 'Focal', 'Dice'], default='NLL')
map_generalizer.add_argument('-reg_loss', '--reg_loss', type=str, help='Specify loss function', choices =['MSE', 'MAE', 'Huber'], default='MAE')

params = map_generalizer.parse_args().__dict__

print('Task is:' + params['task'])
print('Graph convolution is:'  + params['model'])
print('Dropout rate is:' + str(params['dropout']))
data_input = DataInput(data_dir=params['input_dir'], K_orders=params['K_orders'], scales = params['src_tar'])


trainer = MTL_BuildingGen(params=params, data_container=data_input)

if params['mode'] == 'train':
    trainer.train()
    trainer.test()
elif params['mode'] == 'test':
    trainer.test()
else:
    raise NotImplementedError('Invalid mode.')
