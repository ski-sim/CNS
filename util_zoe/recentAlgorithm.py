from util import *
import pandas as pd
from itertools import permutations
from itertools import combinations
import copy
from util_zoe.baseline_dynamicProgramming import *

def originalMerging(bundles_input,bundles_list, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60):
    riders_info = copy.deepcopy(riders_dict)
    bundles_dict = {'Bundle': [], 'Orders': [], 'Reduced Cost': [], 'Rider': []}

    used_list = [set(bundle.shop_seq) for bundle in bundles_list ]

    car_df = bundles_input[bundles_input['Rider'] == 'CAR']
    bike_df = bundles_input[bundles_input['Rider'] == 'BIKE']
    walk_df = bundles_input[bundles_input['Rider'] == 'WALK']

    car_used_df = car_df[ car_df['Orders'].isin(used_list)]
    bike_used_df = bike_df[bike_df['Orders'].isin(used_list)]
    walk_used_df = walk_df[walk_df['Orders'].isin(used_list)]


    dfs = [car_used_df, bike_used_df, walk_used_df]
    rider_types = ['CAR', 'BIKE', 'WALK']

    for itr in range(3):
        df = dfs[itr]
        rider_type = rider_types[itr]
        rider = riders_info[rider_type]

        for bundle1, bundle2 in combinations(df['Bundle'].tolist(), 2):
            merged_orders = set(bundle1.shop_seq + bundle2.shop_seq)
            if len(merged_orders) > 4:
                continue

            original_cost = sum(order_bundles[itr].cost for itr in merged_orders)
            total_volume = get_total_volume(all_orders, merged_orders)
            if total_volume > riders_info[rider_type].capa:
                continue

            reduced_cost = 0
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    if test_feasibility(all_orders, rider, shop_pem, dlv_pem):
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        new_bundle = Bundle(all_orders, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
                        new_reduced_cost = original_cost - new_bundle.cost
                        if new_reduced_cost > reduced_cost:
                            optimal_bundle = new_bundle
                            reduced_cost = new_reduced_cost

            if reduced_cost > 0:
                bundles_dict['Bundle'].append(optimal_bundle)
                bundles_dict['Orders'].append(merged_orders)
                bundles_dict['Reduced Cost'].append(reduced_cost)
                bundles_dict['Rider'].append(rider_type)
    bundles_df = pd.DataFrame(bundles_dict)
    return bundles_df

def jointMerging(bundles_input,bundles_list, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60):
    riders_info = copy.deepcopy(riders_dict)
    bundles_dict = {'Bundle': [], 'Orders': [], 'Reduced Cost': [], 'Rider': []}

    used_list = [set(bundle.shop_seq) for bundle in bundles_list ]

    car_df = bundles_input[bundles_input['Rider'] == 'CAR']
    bike_df = bundles_input[bundles_input['Rider'] == 'BIKE']
    walk_df = bundles_input[bundles_input['Rider'] == 'WALK']

    dfs = [car_df, bike_df, walk_df]
    rider_types = ['CAR', 'BIKE', 'WALK']

    for itr in range(3):
        df = dfs[itr]
        condition1 = df['Reduced Cost'].isin(df.nlargest(20,'Reduced Cost')['Reduced Cost'])
        condition2 = df['Orders'].isin(used_list)
        filtered_df = df.loc[condition1 | condition2]
        rider_type = rider_types[itr]
        rider = riders_info[rider_type]

        for bundle1, bundle2 in combinations(filtered_df['Bundle'].tolist(), 2):
            merged_orders = set(bundle1.shop_seq + bundle2.shop_seq)
            if len(merged_orders) > 4:
                continue

            original_cost = sum(order_bundles[itr].cost for itr in merged_orders)
            total_volume = get_total_volume(all_orders, merged_orders)
            if total_volume > riders_info[rider_type].capa:
                continue

            reduced_cost = 0
            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    if test_feasibility(all_orders, rider, shop_pem, dlv_pem):
                        total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                        new_bundle = Bundle(all_orders, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
                        new_reduced_cost = original_cost - new_bundle.cost
                        if new_reduced_cost > reduced_cost:
                            optimal_bundle = new_bundle
                            reduced_cost = new_reduced_cost

            if reduced_cost > 0:
                bundles_dict['Bundle'].append(optimal_bundle)
                bundles_dict['Orders'].append(merged_orders)
                bundles_dict['Reduced Cost'].append(reduced_cost)
                bundles_dict['Rider'].append(rider_type)

    bundles_df = pd.DataFrame(bundles_dict)
    return bundles_df
