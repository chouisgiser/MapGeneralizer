# -*- coding: utf-8 -*-
"""
# @time    : 28.04.22 21:16
# @author  : zhouzy
# @file    : preprocessing.py
"""
import math

import numpy as np
from shapely.geometry import LineString, Polygon, Point, LinearRing
from shapely.ops import triangulate, split
import geopandas as gpd
import pandas as pd
from collections import defaultdict, Counter
import open3d as o3d
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from utils import label_check, label_check2
import random
from shapely import affinity

def selection(polygon_num):
    source = '../../data/Stuttgart/geb_utm_sp.shp'
    target = '../../data/Stuttgart/geb10.shp'

    df_source = gpd.read_file(source)
    df_target = gpd.read_file(target)
    if 'ORIG_FID' in df_source.columns.tolist():
        df_source.rename(columns = {'ORIG_FID':'JOINID'}, inplace = True)
    df_source_selected = gpd.GeoDataFrame(columns=df_source.columns.tolist())
    df_target_selected = gpd.GeoDataFrame(columns=df_target.columns.tolist())

    df_target = df_target.sort_values('JOINID', ignore_index=True)
    for target_idx, target_row in df_target.iterrows():
        if target_row['geometry'].type == 'Polygon' and df_source_selected.shape[0] < polygon_num and \
                len(target_row['geometry'].interiors) == 0:
            joinid = target_row['JOINID']
            # df_source_row = df_source.loc[df_source['ORIG_FID'] == joinid]
            df_source_row = df_source.loc[df_source['JOINID'] == joinid]
            if df_source_row.shape[0] > 0:
                series_source_row = df_source_row.iloc[0, :]
                if target_row['geometry'].is_valid and series_source_row['geometry'].is_valid:
                    if 0.9 < (target_row['geometry'].area / series_source_row['geometry'].area) <= 1.0 and \
                            series_source_row['geometry'].intersects(target_row['geometry']):
                        df_target_selected = df_target_selected.append(target_row, ignore_index=True)
                        df_source_selected = df_source_selected.append(series_source_row, ignore_index=True)

    df_source_selected.set_crs(crs=df_source.crs)
    df_target_selected.set_crs(crs=df_target.crs)
    df_source_selected.to_file('../../data/MapGeneralizer/selection/geb_5_selected.shp')
    df_target_selected.to_file('../../data/MapGeneralizer/selection/geb_10_selected.shp')
    # df_source_selected.to_file('../../data/MapGeneralizer/selection/geb_10_selected.shp')
    # df_target_selected.to_file('../../data/MapGeneralizer/selection/geb_15_selected.shp')


def get_graph_features(source_row, vertex_list, k):
    # coord_lon_list = list()
    # coord_lat_list = list()
    loc_tri_area_ratio_list = list()
    pre_seg_axis_ratio_list = list()
    next_seg_axis_ratio_list = list()
    dis2mbr_list = list()
    loc_seg_axis_ratio_list = list()
    offset_ratio_list = list()
    loc_seg_length_ratio_list = list()

    reg_tri_area_ratio_list = list()
    reg_semi_per_ratio_list = list()
    reg_radius_ratio_list = list()

    reg_tri_area_list = list()
    reg_semi_per_list = list()
    reg_radius_list = list()
    reg_seg_length_list = list()

    loc_tri_area_list = list()
    loc_turn_angle_list = list()
    loc_convexity_list = list()
    pre_seg_length_list = list()
    next_seg_length_list = list()
    loc_seg_length_list = list()
    offset1_list = list()
    offset2_list = list()
    reg_turn_angle_list = list()
    reg_convexity_list = list()
    pre_seg_ori_list = list()
    next_seg_ori_list = list()

    S0 = source_row['geometry'].area
    L0 = source_row['geometry'].length

    for idx in range(len(vertex_list)):
        vertex = vertex_list[idx]
        pre_vertex_k = vertex_list[(idx - k) % len(vertex_list)]
        next_vertex_k = vertex_list[(idx + k) % len(vertex_list)]

        triangle_abc = Polygon([vertex,  pre_vertex_k, next_vertex_k, vertex])
        pre_seg_length = get_segment_length(pre_vertex_k, vertex)
        next_seg_length = get_segment_length(vertex, next_vertex_k)
        loc_seg_length = get_segment_length(pre_vertex_k, next_vertex_k)
        rect_bound = source_row['geometry'].bounds
        pre_seg_ori = get_orientation(rect_bound, vertex, pre_vertex_k)
        next_seg_ori = get_orientation(rect_bound, vertex, next_vertex_k)

        loc_convexity, loc_turn_angle = get_turning_angle(vertex, pre_vertex_k, next_vertex_k)
        offset1 = get_offset(vertex, pre_vertex_k, vertex_list[(idx + 2) % len(vertex_list)])
        offset2 = get_offset(vertex, vertex_list[(idx - 2) % len(vertex_list)], next_vertex_k)

        # offset_ratio = offset/(S0 ** 0.5)
        reg_seg_length = source_row['geometry'].centroid.distance(Point(vertex))

        coords_o = (source_row['geometry'].centroid.x, source_row['geometry'].centroid.y)
        triangle_obc = Polygon([coords_o,  pre_vertex_k, next_vertex_k, coords_o])
        reg_semi_per = get_semi_perimeter(coords_o, pre_vertex_k, next_vertex_k)
        reg_radius = triangle_obc.area/reg_semi_per

        reg_convexity, reg_turn_angle = get_rotation_angle(source_row['geometry'].centroid, pre_vertex_k, next_vertex_k)


        loc_tri_area_list.append(triangle_abc.area)
        loc_turn_angle_list.append(loc_turn_angle)
        loc_convexity_list.append(loc_convexity)
        pre_seg_length_list.append(pre_seg_length)
        next_seg_length_list.append(next_seg_length)
        pre_seg_ori_list.append(pre_seg_ori)
        next_seg_ori_list.append(next_seg_ori)
        loc_seg_length_list.append(loc_seg_length)
        reg_seg_length_list.append(reg_seg_length)
        offset1_list.append(offset1)
        offset2_list.append(offset2)

        reg_tri_area_list.append(triangle_obc.area)
        reg_semi_per_list.append(reg_semi_per)
        reg_radius_list.append(reg_radius)
        reg_turn_angle_list.append(reg_turn_angle)
        reg_convexity_list.append(reg_convexity)

    df_data = {
        'loc_turn_angle': loc_turn_angle_list,
        'loc_convexity': loc_convexity_list,
        'pre_seg_length': pre_seg_length_list,
        'next_seg_length': next_seg_length_list,
        'loc_tri_area': loc_tri_area_list,
        'loc_seg_length': loc_seg_length_list,
        'reg_tri_area': reg_tri_area_list,
        'reg_semi_per': reg_semi_per_list,
        'reg_radius': reg_radius_list,
        'reg_turn_angle': reg_turn_angle_list,
    }

    df_features = pd.DataFrame(df_data)
    return df_features

def get_offset(vertex, pre_vertex_k, next_vertex_k):
    segment = LineString([pre_vertex_k, next_vertex_k])
    offset = Point(vertex).distance(segment)
    return offset

def get_segment_length (pre_vertex_k, next_vertex_k):
    segment = LineString([pre_vertex_k, next_vertex_k])
    return segment.length

def get_semi_perimeter (vertex, pre_vertex_k, next_vertex_k):
    polygon = Polygon([vertex,  pre_vertex_k, next_vertex_k, vertex])
    semi_perimeter = polygon.length
    return semi_perimeter

def get_dis2mbr(vertex, polygon):
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    min_dis = float('inf')
    max_dis = float('-inf')
    for i in range(len(coords) - 1):
        dis = Point(vertex).distance(Point(coords[i]))
        if min_dis > dis:
            min_dis = dis
        if max_dis < dis:
            max_dis = dis
    dis2mbr_ratio = 1 - min_dis/max_dis

    return min_dis

def get_orientation(bound, vertex, next_vertex):
    minx = bound[0]
    miny = bound[1]
    maxx = bound[2]

    vec_pre = (maxx - minx, 0)
    vec_next = (next_vertex[0] - vertex[0], next_vertex[1] - vertex[1])
    vec_pre_mod = abs(maxx-minx)
    vec_next_mod = LineString([vertex, next_vertex]).length
    cos_value = float("{:.5f}".format((vec_pre[0] * vec_next[0] + vec_pre[1] * vec_next[1]) / (vec_pre_mod * vec_next_mod)))

    return math.acos(cos_value)

def get_effective_area(idx, vertex, vertex_list, polygon):
    total_area = 0
    triangle_abc = Polygon(
        [vertex_list[(idx - 2) % len(vertex_list)], vertex_list[(idx - 1) % len(vertex_list)], vertex])
    intersect_area = triangle_abc.intersection(polygon).area
    if intersect_area > 0.1:
        total_area += intersect_area

    triangle_abc = Polygon(
        [vertex_list[(idx - 1) % len(vertex_list)], vertex, vertex_list[(idx + 1) % len(vertex_list)]])
    intersect_area = triangle_abc.intersection(polygon).area
    if intersect_area > 0.1:
        total_area += intersect_area

    triangle_abc = Polygon(
        [vertex, vertex_list[(idx + 1) % len(vertex_list)], vertex_list[(idx + 2) % len(vertex_list)]])
    intersect_area = triangle_abc.intersection(polygon).area
    if intersect_area > 0.1:
        total_area += intersect_area

    return total_area

def get_turning_angle (vertex, pre_vertex_k, next_vertex_k):

    vec_pre2vertex = (vertex[0] - pre_vertex_k[0], vertex[1] - pre_vertex_k[1])
    vec_vertex2next = (next_vertex_k[0] - vertex[0], next_vertex_k[1] - vertex[1])
    cross_product = vec_pre2vertex[0] * vec_vertex2next[1] - vec_pre2vertex[1] * vec_vertex2next[0]

    vec_pre_mod = LineString([vertex, pre_vertex_k]).length
    vec_next_mod = LineString([vertex, next_vertex_k]).length
    if vec_pre_mod == 0 or vec_next_mod == 0:
        angle_degree = 0
    else:
        cos_value = float("{:.5f}".format((vec_pre2vertex[0] * vec_vertex2next[0] + vec_pre2vertex[1] * vec_vertex2next[1]) / (vec_pre_mod * vec_next_mod)))
        angle_degree = math.degrees(math.acos(cos_value))

    if cross_product > 0:
        sign = 1
    else:
        sign = -1
    return sign, math.radians(angle_degree)

def get_rotation_angle (centrioid, source_coord, target_coord):
    vec_pre = (source_coord[0] - centrioid.x, source_coord[1] - centrioid.y)
    vec_next = (target_coord[0] - centrioid.x, target_coord[1] - centrioid.y)
    vec_pre_mod = LineString([centrioid, source_coord]).length
    vec_next_mod = LineString([centrioid, target_coord]).length
    cos_value = float("{:.5f}".format((vec_pre[0] * vec_next[0] + vec_pre[1] * vec_next[1]) / (vec_pre_mod * vec_next_mod)))
    angle_degree = math.degrees(math.acos(cos_value))

    cross_product = vec_pre[0] * vec_next[1] - vec_pre[1] * vec_next[0]
    if cross_product > 0:
        sign = 1
    else:
        sign = -1
    return sign, math.radians(angle_degree)

def get_long_short_axis(polygon):
    rect = polygon.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    seg1 = LineString([coords[0], coords[1]])
    seg2 = LineString([coords[1], coords[2]])
    if seg1.length < seg2.length:
        long_vec = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        short_vec = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    else:
        long_vec = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        short_vec = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])

    return long_vec, short_vec

def get_axis_ratio(vertex, neighbor_vertex, long_vec, short_vec):
    long_vec_mod = math.sqrt(long_vec[0] ** 2 + long_vec[1] ** 2)
    short_vec_mod = math.sqrt(short_vec[0] ** 2 + short_vec[1] ** 2)

    vertex_vec = (neighbor_vertex[0] - vertex[0], neighbor_vertex[1] - vertex[1])
    vertex_vec_mod = LineString([vertex, neighbor_vertex]).length

    short_cos_value = float(
        "{:.5f}".format((vertex_vec[0] * short_vec[0] + vertex_vec[1] * short_vec[1]) / (vertex_vec_mod * short_vec_mod)))
    short_angle_sin = math.sin(math.acos(short_cos_value))

    long_cos_value = float(
        "{:.5f}".format((vertex_vec[0] * long_vec[0] + vertex_vec[1] * long_vec[1]) / (vertex_vec_mod * long_vec_mod)))
    long_angle_sin = math.sin(math.acos(long_cos_value))

    if short_angle_sin < long_angle_sin:
        ratio = vertex_vec_mod/short_vec_mod
    else:
        ratio = vertex_vec_mod/long_vec_mod

    return ratio

def exclude_nearest_cor(poly_source_coords, poly_target_coords, cor_set, repeated_indices):
    dis_list = list()
    for idx in repeated_indices:
        cor = cor_set[idx, :]
        dis_list.append(Point(poly_source_coords[cor[0]]).distance(Point(poly_target_coords[cor[1]])))
    min_index = dis_list.index(min(dis_list))
    repeated_indices.pop(min_index)
    return repeated_indices

def get_move_vector(cur_src_vertex, pre_src_vertex, next_src_vertex, cur_tar_vertex):
    vec_pre = ( cur_src_vertex[0] - pre_src_vertex[0], cur_src_vertex[1] - pre_src_vertex[1] )
    vec_next = (next_src_vertex[0] - cur_src_vertex[0], next_src_vertex[1] - cur_src_vertex[1])
    vec_tar = (cur_tar_vertex[0] - cur_src_vertex[0], cur_tar_vertex[1] - cur_src_vertex[1])

    vec_pre_mod = LineString([pre_src_vertex, cur_src_vertex]).length
    vec_next_mod = LineString([next_src_vertex, cur_src_vertex]).length

    pre_move = (vec_pre[0] * vec_tar[0] + vec_pre[1] * vec_tar[1])/vec_pre_mod
    next_move = (vec_next[0] * vec_tar[0] + vec_next[1] * vec_tar[1])/vec_next_mod

    return pre_move, next_move

def annotation(poly_source, poly_target, cor_set):

    sf_labels = list()
    move_labels = list()
    pre_move_labels = list()
    next_move_labels = list()

    poly_source_coords = poly_source.exterior.coords[:-1]
    poly_source_centroid = poly_source.centroid
    poly_target_coords = poly_target.exterior.coords[:-1]
    # cor_set = cor_set[:-1, :]

    for idx in range(len(poly_source_coords)):
        if idx in cor_set[:, 0]:
            sf = 1
        else:
            sf = 0
        sf_labels.append(sf)
        # move_labels.append(0)
    target_cors = list()
    pre_target_index = -1
    for idx in range(len(cor_set)):
        if cor_set[idx][1] in target_cors and cor_set[idx][1] < pre_target_index:
            cor_set[idx][1] = pre_target_index
        pre_target_index = cor_set[idx][1]
        target_cors.append(cor_set[idx][1])

    sf_indices = list()
    elements = set(cor_set[:, 1].tolist())
    for element in elements:
        repeated_indices = np.where(cor_set[:, 1] == element)[0]
        if len(repeated_indices) > 1:
            repeated_indices = repeated_indices.tolist()
            repeated_indices = exclude_nearest_cor(poly_source_coords, poly_target_coords, cor_set, repeated_indices)
            repeated_indices = cor_set[repeated_indices]
            sf_indices.extend(repeated_indices[:, 0].tolist())

    for idx in sf_indices:
        sf_labels[idx] = 0

    diff_set = set([idx for idx in range(len(poly_target_coords))]).difference(set(target_cors))

    for cor_target in diff_set:
        pre_cor_idx = -1
        cur_cor_idx = -1
        for idx in range(len(cor_set)):
            if cor_set[idx][1] < cor_set[idx - 1][1]:
                if cor_target > cor_set[idx - 1][1] or cor_target < cor_set[idx][1]:
                    pre_cor_idx = idx - 1
                    cur_cor_idx = idx
                    break
            else:
                if cor_set[idx - 1][1] < cor_target < cor_set[idx][1]:
                    pre_cor_idx = idx - 1
                    cur_cor_idx = idx
                    break
        nearest_source = -1
        min_dis = float("inf")
        pre_cor = cor_set[pre_cor_idx]
        cur_cor = cor_set[cur_cor_idx]
        if pre_cor[0] > cur_cor[0]:
            start = pre_cor[0] - len(poly_source_coords) + 1
        else:
            start = pre_cor[0] + 1
        for cor_source in range(start, cur_cor[0]%len(poly_source_coords)):
            dis_st = Point(poly_target_coords[cor_target]).distance(Point(poly_source_coords[cor_source]))
            if dis_st < min_dis:
                nearest_source = (cor_source + len(poly_source_coords)) % len(poly_source_coords)
                min_dis = dis_st
        sf_labels[nearest_source] = 1
        cor_set = np.insert(cor_set, cur_cor_idx, [nearest_source, cor_target], axis=0)

    cor_sources = cor_set[:, 0].tolist()
    for idx in range(len(poly_source_coords)):
        source_coord = poly_source_coords[idx]
        if idx in cor_sources:
            target_coord = poly_target_coords[cor_set[cor_sources.index(idx), :][1]]
        else:
            target_coord = get_neighbor_target(source_coord, poly_target_coords).coords[0]

        pre_move, next_move = get_move_vector(source_coord, poly_source_coords[idx - 1], poly_source_coords[(idx + 1) % len(poly_source_coords)] , target_coord)
        if sf_labels[idx] == 1:
            if (abs(pre_move) + abs(next_move)) > 0.1:
                sf_labels[idx] = 2


        pre_move_labels.append(pre_move)
        next_move_labels.append(next_move)

    data = {'simplified': sf_labels,
            # 'moved': move_labels,
            'pre_move': pre_move_labels,
            'next_move': next_move_labels}
    df_labels = pd.DataFrame(data)

    return df_labels

def get_neighbor_target(source_coord, poly_target_coords):
    poly_line = LinearRing(poly_target_coords)
    nearest_dis = poly_line.project(Point(source_coord))
    nearest_pt = poly_line.interpolate(nearest_dis)

    return nearest_pt

def alignment(source_row, df_target):
    joinid = source_row['JOINID']
    target_row = df_target.loc[df_target['JOINID'] == joinid].iloc[0, :]

    source_row_geom = source_row['geometry']
    source_xyz = np.zeros((len(source_row_geom.exterior.coords) - 1, 3))
    source_xyz[:, 0] = np.reshape(source_row_geom.exterior.coords.xy[0][:-1], -1)
    source_xyz[:, 1] = np.reshape(source_row_geom.exterior.coords.xy[1][:-1], -1)
    source_xyz[:, 2] = np.reshape((len(source_row_geom.exterior.coords) - 1) * [0], -1)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    source_pcd.paint_uniform_color([1, 0.706, 0])

    target_row_geom = target_row['geometry']
    target_xyz = np.zeros((len(target_row_geom.exterior.coords) - 1, 3))
    target_xyz[:, 0] = np.reshape(target_row_geom.exterior.coords.xy[0][:-1], -1)
    target_xyz[:, 1] = np.reshape(target_row_geom.exterior.coords.xy[1][:-1], -1)
    target_xyz[:, 2] = np.reshape((len(target_row_geom.exterior.coords) - 1) * [0], -1)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
    target_pcd.paint_uniform_color([0, 0.651, 0.929])

    threshold = 5
    trans_init = np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    reg_p2p = o3d.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=5000))
    # print(reg_p2p)
    cor_set = np.asarray(reg_p2p.correspondence_set)

    return target_row, cor_set

bounding_x_list = list()
bounding_y_list = list()

def polygon2graph(source_row):
    df_vertexs = pd.DataFrame(columns=['osm_id', 'vid', 'coord_lon', 'coord_lat'])
    adj_dict = defaultdict(list)
    adj_attr = dict()

    vertex_list = list()
    vid_list = list()
    lon_list = list()
    lat_list = list()
    norm_lon_list = list()
    norm_lat_list = list()

    polygon = source_row['geometry']
    centroid_x = polygon.centroid.x
    centroid_y = polygon.centroid.y
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):
        vertex_list.append(coords[i])
    for i in range(len(vertex_list)):
        vid_list.append(i)
        lon_list.append(vertex_list[i][0])
        lat_list.append(vertex_list[i][1])

        norm_lon_list.append(vertex_list[i][0] - centroid_x)
        norm_lat_list.append(vertex_list[i][1] - centroid_y)

        adj_dict[i] = [(i - 1 + len(vertex_list)) % len(vertex_list), (i + 1) % len(vertex_list) ]

        pre_id = (i - 1 + len(vertex_list)) % len(vertex_list)
        next_id = (i + 1) % len(vertex_list)
        adj_attr[(i, pre_id)] = {"distance": Point(coords[i]).distance(Point(coords[pre_id]))}
        adj_attr[(i, next_id)] = {"distance": Point(coords[i]).distance(Point(coords[next_id]))}

    df_vertexs['osm_id'] = [int(source_row['JOINID'])] * len(vertex_list)
    df_vertexs['vid'] = vid_list
    df_vertexs['coord_lon'] = lon_list
    df_vertexs['coord_lat'] = lat_list
    df_vertexs['coord_lon_norm'] = norm_lon_list
    df_vertexs['coord_lat_norm'] = norm_lat_list

    bounding_x_list.append(max(norm_lon_list))
    bounding_x_list.append(min(norm_lon_list))
    bounding_y_list.append(max(norm_lat_list))
    bounding_y_list.append(min(norm_lat_list))

    return df_vertexs, adj_dict, adj_attr


def ply_reflection(polygon, seg_axis):
    reflect_coords = list()
    ply_coords = list(polygon.exterior.coords)
    for coord in ply_coords:
        inter_dis = seg_axis.project(Point(coord))
        mid_point = seg_axis.interpolate(inter_dis)
        reflect_coord = (2 * mid_point.x - coord[0], 2 * mid_point.y - coord[1])
        reflect_coords.append(reflect_coord)
    reflect_coords.reverse()
    reflect_ply = Polygon(reflect_coords)
    return reflect_ply

def graph_building(source_file, target_file, K_orders):
    map_adj_list = list()
    labels_list = list()
    vertex_info_list = list()
    # edge_attr_list = list()

    df_total_labels = pd.DataFrame(columns=['osm_id', 'simplified', 'pre_move', 'next_move'])

    df_source = gpd.read_file(source_file)
    df_target = gpd.read_file(target_file)

    df_source_selected = df_source
    for idx, source_row in df_source_selected.iterrows():

        if source_row['JOINID'] not in df_target['JOINID'].tolist():
            continue
        if len(source_row['geometry'].exterior.coords) > 33:
            continue
        print(source_row['JOINID'])
        df_vertexs, adj_dict, adj_attr = polygon2graph(source_row)
        # print(df_vertexs.shape)
        target_row, cor_set = alignment(source_row, df_target)

        # iou = source_row['geometry'].intersection(target_row['geometry']).area/source_row['geometry'].union(target_row['geometry']).area
        # if iou >= 0.99:
        #     continue

        G = nx.from_dict_of_lists(adj_dict)
        graph_adj_dict = nx.adjacency_matrix(G)


        df_source_row_labels = annotation(source_row['geometry'], target_row['geometry'], cor_set)
        labels_list.append(df_source_row_labels.to_numpy())

        df_source_row_labels = pd.concat([pd.DataFrame(data={'osm_id': [source_row['JOINID']] * df_source_row_labels.shape[0]}),
                                    df_source_row_labels], axis=1)
        df_total_labels = df_total_labels.append(df_source_row_labels, ignore_index=True)

        vertex_list = [(row['coord_lon'], row['coord_lat']) for vertex_idx, row in df_vertexs.iterrows()]

        K_orders_features = pd.DataFrame()
        for i in range(len(K_orders)):
            k = K_orders[i]
            k_features_columns = ['loc_turn_angle_{}'.format(str(k)),'loc_convexity_{}'.format(str(k)),
                                    'pre_seg_length_{}'.format(str(k)), 'next_seg_length_{}'.format(str(k)),
                                    'loc_tri_area_{}'.format(str(k)), 'loc_seg_length_{}'.format(str(k)),
                                    'reg_tri_area_{}'.format(str(k)), 'reg_semi_per_{}'.format(str(k)),
                                    'reg_radius_{}'.format(str(k)), 'reg_turn_angle_{}'.format(str(k)),
                                    ]
            df_features = pd.DataFrame(columns=k_features_columns)
            df_features[k_features_columns] = get_graph_features(source_row, vertex_list, k)

            K_orders_features = pd.concat([K_orders_features, df_features], axis=1)

        vertex_info = pd.concat([df_vertexs, K_orders_features, df_source_row_labels.iloc[:, 1:4]], axis = 1)
        vertex_info_list.append(vertex_info)
        map_adj_list.append(graph_adj_dict)

    df_np_vertex_info = pd.concat(vertex_info_list, axis=0)
    df_np_vertex_info['coord_lon_norm'] = df_np_vertex_info['coord_lon_norm']/np.std(bounding_x_list )
    df_np_vertex_info['coord_lat_norm'] = df_np_vertex_info['coord_lat_norm']/np.std(bounding_y_list )
    scale_feature_columns = []
    for i in  range(len(K_orders)):
        k = K_orders[i]
        scale_feature_columns.extend(['loc_turn_angle_{}'.format(str(k)),
                                    'pre_seg_length_{}'.format(str(k)), 'next_seg_length_{}'.format(str(k)),
                                    'loc_tri_area_{}'.format(str(k)), 'loc_seg_length_{}'.format(str(k)),
                                    'reg_tri_area_{}'.format(str(k)), 'reg_semi_per_{}'.format(str(k)),
                                    'reg_radius_{}'.format(str(k)), 'reg_turn_angle_{}'.format(str(k)),
                                    ])
    ss = StandardScaler()
    df_np_vertex_info[scale_feature_columns] = ss.fit_transform(df_np_vertex_info[scale_feature_columns])
    vertex_info_list.clear()
    groups = df_np_vertex_info.groupby('osm_id', sort=False)
    for name, group in groups:
        vertex_info_list.append(group.to_numpy())

    np_vertex_info = np.asarray(vertex_info_list, dtype=object)
    np_adj_info = np.asarray(map_adj_list, dtype=object)

    print(len(np_vertex_info))
    orders = ''
    for i in range(len(K_orders)):
        orders += str(K_orders[i])
        if i < len(K_orders) - 1:
            orders += '&'

    mode = source_file.split('_')[4].split('.')[0]
    np.save('../../data/MapGeneralizer/input/vertex_{}.npy'.format(mode), np_vertex_info)
    np.save('../../data/MapGeneralizer/input/adj_{}.npy'.format(mode), np_adj_info)
    return df_total_labels

def get_inflection_angle (vertex, pre_vertex_k, next_vertex_k):
    vec_pre = (pre_vertex_k[0] - vertex[0], pre_vertex_k[1] - vertex[1])
    vec_next = (next_vertex_k[0] - vertex[0], next_vertex_k[1] - vertex[1])
    vec_pre_mod = LineString([vertex, pre_vertex_k]).length
    vec_next_mod = LineString([vertex, next_vertex_k]).length
    if vec_pre_mod == 0 or vec_next_mod == 0:
        angle_degree = 0
    else:
        cos_value = float(
            "{:.5f}".format((vec_pre[0] * vec_next[0] + vec_pre[1] * vec_next[1]) / (vec_pre_mod * vec_next_mod)))
        angle_degree = math.degrees(math.acos(cos_value))

    vec_pre2vertex = (vertex[0] - pre_vertex_k[0], vertex[1] - pre_vertex_k[1])
    vec_vertex2next = (next_vertex_k[0] - vertex[0], next_vertex_k[1] - vertex[1])
    cross_product = vec_pre2vertex[0] * vec_vertex2next[1] - vec_pre2vertex[1] * vec_vertex2next[0]
    if cross_product > 0:
        sign = 1
    else:
        sign = -1

    return sign, angle_degree

def geometry_cleaning(file, threshold_dis, threshold_angle):
    df_file = gpd.read_file(file)
    clean_geom_list = list()
    for idx, source_row in df_file.iterrows():
        ply_coords = list(source_row['geometry'].exterior.coords)
        cleaned_ids = set()
        for i in range(0, len(ply_coords) - 1):
            j = i + 1
            distance = Point(ply_coords[i]).distance(Point(ply_coords[j]))
            if distance <= threshold_dis:
                if j == len(ply_coords) - 1:
                    cleaned_ids.add(i)
                else:
                    cleaned_ids.add(j)

            _, angle = get_inflection_angle(ply_coords[j], ply_coords[i], ply_coords[(j + 1) % len(ply_coords)])
            if angle <= threshold_angle:
                if j != len(ply_coords) - 1:
                    cleaned_ids.add(j)


        ply_coords = [ply_coords[i] for i in range(len(ply_coords)) if i not in cleaned_ids]
        if len(ply_coords) > 2:
            cleaned_geom = Polygon(ply_coords)
            iou = cleaned_geom.intersection(source_row['geometry']).area / cleaned_geom.union(source_row['geometry']).area
            if iou < 0.99:
                print('{} iou is: {}'.format(source_row['JOINID'], iou))

            clean_geom_list.append(cleaned_geom)
        else:
            print('{} polygon is not valid'.format(source_row['JOINID']))

    df_file['geometry'] = clean_geom_list

    df_file.to_file(file.replace('.shp', '_clean.shp'))

def dataset_split(source_file, target_file, train_ratio, valid_ratio, test_ratio):
    df_source = gpd.read_file(source_file)
    df_target = gpd.read_file(target_file)
    src_simplified_indices = list()
    tar_simplified_indices = list()
    src_no_simplified_indices = list()
    tar_no_simplified_indices = list()

    for idx, source_row in df_source.iterrows():
        joinid = source_row['JOINID']
        target_row = df_target.loc[df_target['JOINID'] == joinid].iloc[0, :]
        iou = source_row['geometry'].intersection(target_row['geometry']).area/source_row['geometry'].union(target_row['geometry']).area
        if iou < 0.99:
            src_simplified_indices.append(idx)
            target_row_idx = df_target.index[df_target['JOINID'] == joinid].tolist()[0]
            tar_simplified_indices.append(target_row_idx)
        else:
            if len(src_no_simplified_indices) < len(src_simplified_indices):
                src_no_simplified_indices.append(idx)
                target_row_idx = df_target.index[df_target['JOINID'] == joinid].tolist()[0]
                tar_no_simplified_indices.append(target_row_idx)

    src_indices = src_simplified_indices + src_no_simplified_indices
    tar_indices = tar_simplified_indices + tar_no_simplified_indices
    df_source_simplified = df_source.iloc[src_indices].reset_index(drop=True)
    df_target_simplified = df_target.iloc[tar_indices].reset_index(drop=True)
    print(df_source_simplified.shape)


    total_indices = [i for i in range(df_source_simplified.shape[0])]
    np.random.shuffle(total_indices)
    train_size = int(df_source_simplified.shape[0] * train_ratio / (train_ratio + valid_ratio + test_ratio))
    valid_size = int(df_source_simplified.shape[0] * valid_ratio / (train_ratio + valid_ratio + test_ratio))

    train_indices = total_indices[:train_size]
    print(len(train_indices))
    valid_indices = total_indices[train_size: train_size + valid_size]
    print(len(valid_indices))
    test_indices = total_indices[train_size + valid_size:]
    print(len(test_indices))
    src_train_set = df_source_simplified.iloc[train_indices]
    tar_train_set = df_target_simplified.iloc[train_indices]
    src_valid_set = df_source_simplified.iloc[valid_indices]
    tar_valid_set = df_target_simplified.iloc[valid_indices]
    src_test_set = df_source_simplified.iloc[test_indices]
    tar_test_set = df_target_simplified.iloc[test_indices]

    print(src_train_set.shape)
    print(src_valid_set.shape)
    print(src_test_set.shape)

    src_train_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_5_selected_train.shp')
    tar_train_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_10_selected_train.shp')
    src_valid_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_5_selected_valid.shp')
    tar_valid_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_10_selected_valid.shp')
    src_test_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_5_selected_test.shp')
    tar_test_set.to_file('../../data/MapGeneralizer/selection/dataset/geb_10_selected_test.shp')

# clean redunant vertices along the polygonal boundaries
# source_file = '../../data/MapGeneralizer/selection/geb_15_selected.shp'
# geometry_cleaning(source_file, threshold_dis=0.01, threshold_angle=1)

# preprocessing source maps and target maps
# selection(175756)

# build training, validation, and test buildings with 6:2:2
# source_file = '../../data/MapGeneralizer/selection/geb_5_selected.shp'
# target_file = '../../data/MapGeneralizer/selection/geb_10_selected.shp'
# dataset_split(source_file, target_file, 6, 2, 2)

# build numpy files for training, validation, and test set
src_train_file = '../../data/MapGeneralizer/selection/dataset/geb_5_selected_train.shp'
tar_train_file = '../../data/MapGeneralizer/selection/dataset/geb_10_selected_train.shp'
df_train = graph_building(src_train_file, tar_train_file, [1])

src_valid_file = '../../data/MapGeneralizer/selection/bdataset/geb_5_selected_valid.shp'
tar_valid_file = '../../data/MapGeneralizer/selection/dataset/geb_10_selected_valid.shp'
df_valid = graph_building(src_valid_file, tar_valid_file, [1])

src_test_file = '../../data/MapGeneralizer/selection/dataset/geb_5_selected_test.shp'
tar_test_file = '../../data/MapGeneralizer/selection/dataset/geb_10_selected_test.shp'
df_test = graph_building(src_test_file, tar_test_file, [1])
