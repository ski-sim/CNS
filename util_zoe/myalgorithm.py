import copy
import pandas as pd
from util import *
from util_zoe.baseline_dynamicProgramming import *
from util_zoe.topTen import *
from util_zoe.recentAlgorithm import *
from util_zoe.knapsack import *
from util_LNS import *
from pastalgorithm import *

def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r

    initial_df, order_bundles = initialMerging(K, all_orders, riders_dict, dist_mat, timelimit=60)
    initial_list = bundleOptimization(initial_df, K, riders_dict)
    best_obj = sum((bundle.cost for bundle in initial_list)) / K
    print(f'Best obj = {best_obj}')
    opt_bundles = initial_list

    binary_df = binaryMerging(initial_df, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)
    binary_appended_df = pd.concat([binary_df,initial_df],ignore_index=True)
    binary_list = bundleOptimization(binary_appended_df, K, riders_dict)
    

    cur_obj = sum((bundle.cost for bundle in binary_list)) / K
    if cur_obj < best_obj:
        best_obj = cur_obj
        print(f'Best obj = {best_obj}')
        opt_bundles = binary_list
    

    
    # Real Algorithm starts from here!


    joint_df = jointMerging(binary_appended_df, binary_list, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)
    joint_append_df = pd.concat([joint_df, binary_appended_df], ignore_index=True)
    joint_list = bundleOptimization(joint_append_df, K, riders_dict)

    cur_obj = sum((bundle.cost for bundle in joint_list)) / K
    if cur_obj < best_obj:
        best_obj = cur_obj
        print(f'Best obj = {best_obj}')
        opt_bundles = joint_list
    
 

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in opt_bundles
    ]
    #------------- End of custom algorithm code--------------#
    return solution
    

