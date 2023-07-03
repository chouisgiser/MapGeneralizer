# -*- coding: utf-8 -*-
"""
# @time    : 05.05.22 11:22
# @author  : zhouzy
# @file    : utils.py
"""
import numpy as np
import torch
import geopandas as gpd
from shapely import affinity
from shapely.geometry import Point, Polygon, LineString, LinearRing
import math
torch.manual_seed(0)
from sympy import solve, nsolve, Symbol, Eq

class DataInput(object):
    def __init__(self, data_dir: str, K_orders: list, scales: list):
        self.data_dir = data_dir
        self.orders = K_orders
        self.scales = scales

    def load_data(self):
        orders = ''
        for i in range(len(self.orders)):
            orders += str(self.orders[i])
            if i < len(self.orders) - 1:
                orders += '&'

        # adj = np.load('{}/adj_{}_{}.npy'.format(self.data_dir, self.scales[0], self.scales[1]), allow_pickle=True)
        # features = np.load('{}/vertex_{}_{}.npy'.format(self.data_dir, self.scales[0], self.scales[1]), allow_pickle=True)
        train_adj = np.load('{}/adj_train.npy'.format(self.data_dir), allow_pickle=True)
        train_features = np.load('{}/vertex_train.npy'.format(self.data_dir), allow_pickle=True)
        
        valid_adj = np.load('{}/adj_valid.npy'.format(self.data_dir), allow_pickle=True)
        valid_features = np.load('{}/vertex_valid.npy'.format(self.data_dir), allow_pickle=True)
        
        test_adj = np.load('{}/adj_test.npy'.format(self.data_dir), allow_pickle=True)
        test_features = np.load('{}/vertex_test.npy'.format(self.data_dir), allow_pickle=True)

        data = dict()
        # data['adj_matrix'] = adj
        # data['features'] = features
        # data['edge_attr'] = edge_attr
        data['train_adj_matrix'] = train_adj
        data['train_features'] = train_features
        data['valid_adj_matrix'] = valid_adj
        data['valid_features'] = valid_features
        data['test_adj_matrix'] = test_adj
        data['test_features'] = test_features

        return data

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def precision(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    TP = correct.sum()
    FP = preds.gt(labels).double()
    FP = FP.sum()
    return TP/(TP + FP)

def recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    TP = correct.sum()
    FN = preds.lt(labels).double()
    FN = FN.sum()
    return TP / (TP + FN)

def MTL_recontruct_points(graph_features, pred_rm, pred_move_dir, pred_move_dis):
    gpd_data = {
        'osm_id': graph_features[:, 0],
        'vid': [vid for vid in range(len(graph_features))],
        'geometry': [Point(graph_features[vid][2], graph_features[vid][3] ) for vid in range(len(graph_features)) ],
        'gt_removal': graph_features[:, -4],
        'gt_dir': graph_features[:, -2],
        'gt_dis': graph_features[:, -1],
        'pred_removal': pred_rm.tolist(),
        'pred_dir': pred_move_dir.tolist(),
        'pred_dis': pred_move_dis.tolist()
    }
    gpd_points = gpd.GeoDataFrame(data=gpd_data )

    return gpd_points

def MTL_recontruct_points2(graph_features, pred_rm, pred_move_dir, pred_move_dis):
    gpd_data = {
        'osm_id': graph_features[:, 0],
        'vid': [vid for vid in range(len(graph_features))],
        'geometry': [Point(graph_features[vid][2], graph_features[vid][3] ) for vid in range(len(graph_features)) ],
        'gt_removal': graph_features[:, -3],
        # 'gt_move': graph_features[:, -3],
        'gt_predis': graph_features[:, -2],
        'gt_nextdis': graph_features[:, -1],
        'pred_removal': pred_rm.tolist(),
        # 'pred_move': pred_mv.tolist(),
        'pred_predis': pred_move_dir.tolist(),
        'pred_nextdis': pred_move_dis.tolist()
    }
    gpd_points = gpd.GeoDataFrame(data=gpd_data )

    return gpd_points
    
def MTL_recontruct_points3(graph_features, pred_move_dir, pred_move_dis):
    gpd_data = {
        'osm_id': graph_features[:, 0],
        'vid': [vid for vid in range(len(graph_features))],
        'geometry': [Point(graph_features[vid][2], graph_features[vid][3] ) for vid in range(len(graph_features)) ],
        'gt_removal': graph_features[:, -3],
        # 'gt_move': graph_features[:, -3],
        'gt_predis': graph_features[:, -2],
        'gt_nextdis': graph_features[:, -1],
        # 'pred_removal': pred_rm.tolist(),
        # 'pred_move': pred_mv.tolist(),
        'pred_predis': pred_move_dir.tolist(),
        'pred_nextdis': pred_move_dis.tolist()
    }
    gpd_points = gpd.GeoDataFrame(data=gpd_data )

    return gpd_points

def STL_recontruct_points(graph_features, pred):
    gpd_data = {
        'osm_id': graph_features[:, 0],
        'geometry': [Point(graph_features[vid][2], graph_features[vid][3] ) for vid in range(len(graph_features)) ],
        'gt_removal': graph_features[:, -4],
        'gt_move_dir': graph_features[:, -2],
        'gt_move_dis': graph_features[:, -1],
        'pred': pred.tolist()
    }
    gpd_points = gpd.GeoDataFrame(data=gpd_data )

    return gpd_points

def label_check(ply_features, polygon):
    ply_id = int(ply_features[0][0])
    source_ply_coords = [(ply_features[vid][2], ply_features[vid][3]) for vid in range(len(ply_features))]
    source_ply_coords.append(source_ply_coords[0])
    source_ply = Polygon(source_ply_coords)
    centroid = source_ply.centroid
    

    target_ply_coords = list()
    for vid in range(len(ply_features)):
        vertex_features = ply_features[vid]
        ref_seg = LineString([centroid, (vertex_features[2], vertex_features[3])])
        if vertex_features[-3] == -1:
            angle = -1 * vertex_features[-2]
        else:
            angle = vertex_features[-2]
        rotated_seg = affinity.rotate(ref_seg, angle, origin=centroid)
        relative_angle = math.atan2(rotated_seg.coords[1][1] - rotated_seg.coords[0][1],
                                    rotated_seg.coords[1][0] - rotated_seg.coords[0][0])
        target_dis = rotated_seg.length + vertex_features[-1]
        delta_x = target_dis * math.cos(relative_angle)
        delta_y = target_dis * math.sin(relative_angle)
        target_coord = (centroid.x + delta_x, centroid.y + delta_y)

        if vertex_features[-4] == 0:
            target_ply_coords.append(target_coord)

    target_ply_coords.append(target_ply_coords[0])
    target_ply = Polygon(target_ply_coords)
    if not polygon.is_valid:
        print(str(ply_id) + " reference polygon is not valid.")
        return False
    elif not target_ply.is_valid:
        print(str(ply_id) + " reconstructed polygon is not valid.")
        return False
    else:
        iou = target_ply.intersection(polygon).area / target_ply.union(polygon).area
        print('polygon {}\'s iou is: {}'.format(ply_id, iou))
        return True

def label_check2(ply_features, polygon):
    ply_id = int(ply_features[0][0])
    source_ply_coords = [(ply_features[vid][2], ply_features[vid][3]) for vid in range(len(ply_features))]
    source_ply_coords.append(source_ply_coords[0])

    target_ply_coords = list()
    for vid in range(len(ply_features)):
        vertex_features = ply_features[vid]
        cur_src_coord = (vertex_features[2], vertex_features[3])
        pre_src_coord = (ply_features[vid - 1][2], ply_features[vid - 1][3])
        next_src_coord = (ply_features[(vid + 1) % len(ply_features)][2], ply_features[(vid + 1) % len(ply_features)][3])
        vec_pre = (cur_src_coord[0] - pre_src_coord[0], cur_src_coord[1] - pre_src_coord[1])
        vec_next = (next_src_coord[0] - cur_src_coord[0], next_src_coord[1] - cur_src_coord[1])
        vec_pre_mod = LineString([pre_src_coord, cur_src_coord]).length
        vec_next_mod = LineString([next_src_coord, cur_src_coord]).length

        # tar_x = Symbol('tar_x')
        # tar_y = Symbol('tar_y')
        # eq_results = solve([Eq(vec_pre[0] * tar_x + vec_pre[1] * tar_y, vec_pre_mod * vertex_features[-2]),
        #         Eq((vec_next[0] * tar_x + vec_next[1] * tar_y), vec_next_mod * vertex_features[-1])], [tar_x, tar_y])
        # target_coord = (cur_src_coord[0] + eq_results[tar_x], cur_src_coord[1] + eq_results[tar_y])
        
        A = np.array([[vec_pre[0], vec_pre[1]], [vec_next[0], vec_next[1]]])
        b = np.array([vec_pre_mod * vertex_features[-2], vec_next_mod * vertex_features[-1]])
        eq_results = np.linalg.solve(A,b)
        target_coord = (cur_src_coord[0] + eq_results[0], cur_src_coord[1] + eq_results[1])

        if vertex_features[-4] == 0:
            target_ply_coords.append(target_coord)

    target_ply_coords.append(target_ply_coords[0])
    target_ply = Polygon(target_ply_coords)
    if not polygon.is_valid:
        print(str(ply_id) + " reference polygon is not valid.", flush=True)
        return False
    elif not target_ply.is_valid:
        print(str(ply_id) + " reconstructed polygon is not valid.", flush=True)
        return False
    else:
        iou = target_ply.intersection(polygon).area / target_ply.union(polygon).area
        print('polygon {}\'s iou is: {}'.format(ply_id, iou), flush=True)
        return True

def reconstruct_polygons2(pt_file, gt_target_file):
    gt_target_plys = gpd.read_file(gt_target_file)
    plys_points = gpd.read_file(pt_file)
    plys_points = plys_points.groupby('osm_id')

    iou_list = list()
    pos_error_list = list()
    pred_target_geoms = list()
    pred_target_ids = list()
    pos_error_id_list = list()
    for name, ply_points in plys_points:
        osm_id = str(int(ply_points.iloc[0]['osm_id']))
        # if osm_id not in ['383']:
        #     continue
        gt_target_ply = gt_target_plys.loc[gt_target_plys['JOINID'] == osm_id].iloc[0, :]['geometry']
        source_ply_coords = [row['geometry'] for idx, row in ply_points.iterrows()]
        source_ply_coords.append(source_ply_coords[0])
        source_ply = Polygon(source_ply_coords)
        centroid = source_ply.centroid

        ref_ply = gt_target_ply

        target_ply_coords = list()
        for idx in range(ply_points.shape[0]):
            row = ply_points.iloc[idx]
            cur_src_pt = row['geometry']
            pre_src_pt = ply_points.iloc[idx-1]['geometry']
            next_src_pt = ply_points.iloc[(idx + 1) % ply_points.shape[0], :]['geometry']
            vec_pre = (cur_src_pt.x - pre_src_pt.x, cur_src_pt.y - pre_src_pt.y)
            vec_next = (next_src_pt.x - cur_src_pt.x, next_src_pt.y - cur_src_pt.y)
            vec_pre_mod = LineString([pre_src_pt, cur_src_pt]).length
            vec_next_mod = LineString([next_src_pt, cur_src_pt]).length
            # tar_x = Symbol('tar_x')
            # tar_y = Symbol('tar_y')
            # eq_results = solve([Eq(vec_pre[0] * tar_x + vec_pre[1] * tar_y, vec_pre_mod * row['pred_dir']),
            #                     Eq((vec_next[0] * tar_x + vec_next[1] * tar_y), vec_next_mod * row['pred_dis'])],
            #                   [tar_x, tar_y])
            # target_coord = (cur_src_pt.x + eq_results[tar_x], cur_src_pt.y + eq_results[tar_y])
            
            try:
                A = np.array([[vec_pre[0], vec_pre[1]], [vec_next[0], vec_next[1]]])
                b = np.array([vec_pre_mod * row['pred_dir'], vec_next_mod * row['pred_dis']])
                eq_results = np.linalg.solve(A,b)
            except np.linalg.LinAlgError:
                target_coord = (cur_src_pt.x, cur_src_pt.y)
            else:
                # print(eq_results)
                target_coord = (cur_src_pt.x + eq_results[0], cur_src_pt.y + eq_results[1])
            finally:
                if row['pred_remov'] == 0:
                    target_ply_coords.append(target_coord)
                    ref_ply_coords = list(ref_ply.exterior.coords)
                    ref_ply_coords.pop()
                    poly_line = LinearRing(ref_ply_coords)
                    nearest_dis = poly_line.project(Point(target_coord))
                    nearest_pt = poly_line.interpolate(nearest_dis)
                    pos_error = nearest_pt.distance(Point(target_coord))
                    pos_error_list.append(pos_error)
                    pos_error_id_list.append(row['osm_id'])

        target_ply_coords.append(target_ply_coords[0])
        if len(target_ply_coords) < 3:
            print(osm_id)
            continue
        target_ply = Polygon(target_ply_coords)
        if not target_ply.is_valid:
            print(osm_id)
            continue
        pred_target_geoms.append(target_ply)
        pred_target_ids.append(osm_id)

        iou = target_ply.intersection(ref_ply).area / target_ply.union(ref_ply).area
        print('polygon {}\'s iou is: {}'.format(osm_id, iou), flush = True)
        iou_list.append(iou)

    gpd_data = {
        'osm_id': pred_target_ids,
        'geometry': pred_target_geoms
    }
    gpd_plys = gpd.GeoDataFrame(data=gpd_data)

    print(pos_error_id_list[pos_error_list.index(max(pos_error_list))])
    print(sum(pos_error_list) / len(pos_error_list))
    print(sum(iou_list) / len(iou_list))
    print(iou_list.index(min(iou_list)))
    gpd_plys.to_file(pt_file.replace('.shp', '_polygon.shp'))
    return gpd_plys, pos_error_list, iou_list

def MTL_reconstruct_polygon(gt_tensor, pred_rm, pred_preMove, pred_nextMove, Y):
    gt_areas = list()
    pred_areas = list()
    
    gt_inangle_sums = list()
    pred_inangle_sums = list()

    previous_ply_id = -1
    gt_coords = list()
    pred_coords = list()
    for idx in range(len(gt_tensor)):
        if gt_tensor[idx][0] != previous_ply_id:
            if previous_ply_id != -1:
                
                gt_inangle_sums.append((len(gt_coords) - 2) * 180)
                pred_inangle_sums.append((len(pred_coords) - 2) * 180)
                
                gt_coords.append(gt_coords[0])
                gt_areas.append(Polygon(gt_coords).area)
                if len(pred_coords) > 2:
                    pred_coords.append(pred_coords[0])
                    pred_areas.append(Polygon(pred_coords).area)  
                else:
                    pred_areas.append(0.0)  
                
            previous_ply_id = gt_tensor[idx][0]
            gt_coords.clear()
            pred_coords.clear()
        
        # gt_coords.append((gt_tensor[idx][2].item(), gt_tensor[idx][3].item()))
        cur_src_coord = (gt_tensor[idx][2].item(), gt_tensor[idx][3].item())
        pre_src_coord = (gt_tensor[idx - 1][2].item(), gt_tensor[idx - 1][3].item())
        next_src_coord = (gt_tensor[(idx + 1) % len(gt_tensor)][2].item(), gt_tensor[(idx + 1) % len(gt_tensor)][3].item())
        vec_pre = (cur_src_coord[0] - pre_src_coord[0], cur_src_coord[1] - pre_src_coord[1])
        vec_next = (next_src_coord[0] - cur_src_coord[0], next_src_coord[1] - cur_src_coord[1])
        vec_pre_mod = LineString([pre_src_coord, cur_src_coord]).length
        vec_next_mod = LineString([next_src_coord, cur_src_coord]).length
        
        # print(Y)
        # print(Y[idx])
        
        if Y[idx][0].item() != 0:
            
            try:
                A = np.array([[vec_pre[0], vec_pre[1]], [vec_next[0], vec_next[1]]])
                b = np.array([vec_pre_mod * Y[idx][1].item(), vec_next_mod * Y[idx][2].item()])
                eq_results = np.linalg.solve(A,b)
                # print(eq_results)
            except np.linalg.LinAlgError:
                gt_coord = (cur_src_coord[0], cur_src_coord[1])
            else:
                gt_coord = (cur_src_coord[0] + eq_results[0], cur_src_coord[1] + eq_results[1])
            finally:
                gt_coords.append(gt_coord)
        
        if pred_rm[idx].item() != 0:
            try:
                A = np.array([[vec_pre[0], vec_pre[1]], [vec_next[0], vec_next[1]]])
                b = np.array([vec_pre_mod * pred_preMove[idx].item(), vec_next_mod * pred_nextMove[idx].item()])
                eq_results = np.linalg.solve(A,b)
            except np.linalg.LinAlgError:
                pred_coord = (cur_src_coord[0], cur_src_coord[1])
            else:
                # print(eq_results)
                pred_coord = (cur_src_coord[0] + eq_results[0], cur_src_coord[1] + eq_results[1])
            finally:
                pred_coords.append(pred_coord)
    
    gt_inangle_sums.append((len(gt_coords) - 2) * 180.0)
    pred_inangle_sums.append((len(pred_coords) - 2) * 180.0)
    
    if len(gt_coords) > 2:
        gt_coords.append(gt_coords[0])
        gt_areas.append(Polygon(gt_coords).area)
    else:
        gt_areas.append(0.0)
    if len(pred_coords) > 2:
        pred_coords.append(pred_coords[0])
        pred_areas.append(Polygon(pred_coords).area)  
    else:
        pred_areas.append(0.0)
    return torch.tensor(gt_areas), torch.tensor(pred_areas), torch.tensor(gt_inangle_sums), torch.tensor(pred_inangle_sums)

def automatic_weight(model, task_loss):
    """
    It is adapted from https://github.com/Mikoto10032/AutomaticWeightedLoss.git
    The orginal paper is: Auxiliary tasks in multi-task learning
    """

    total_loss = 0
    for i in range(len(task_loss)):
        total_loss += 0.5 / (model.weights[i] ** 2) * task_loss[i] + torch.log(1 + model.weights[i] ** 2)
    return total_loss


