"""
# @time    : 23.07.22 21:30
# @author  : zhouzy
# @file    : generalization_eval.py
"""
from utils import reconstruct_polygons2

pt_file = '../../data/MapGeneralizer/output/Bldgs_Gen_prediction.shp'
gt_target_file = '../../data/MapGeneralizer/selection/geb_10_selected.shp'
gpd_plys, pos_error_list, iou_list = reconstruct_polygons2(pt_file, gt_target_file)