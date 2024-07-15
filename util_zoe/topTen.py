from util import *
import pandas as pd
from itertools import permutations
from itertools import combinations
import copy
from util_zoe.baseline_dynamicProgramming import *

def topMerging(bundles_input, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60):
    riders_info = copy.deepcopy(riders_dict)

    new_raw_dict = {'Bundle': [], 'Orders': [], 'Reduced Cost': [], 'Rider': []}
    bundles_df = pd.DataFrame(new_raw_dict)

    car_df = bundles_input[bundles_input['Rider'] == 'CAR']
    bike_df = bundles_input[bundles_input['Rider'] == 'BIKE']
    walk_df = bundles_input[bundles_input['Rider'] == 'WALK']

    car_df_descending = car_df.sort_values(by='Reduced Cost', ascending=False)
    bike_df_descending = bike_df.sort_values(by='Reduced Cost', ascending=False)
    walk_df_descending = walk_df.sort_values(by='Reduced Cost', ascending=False)

    car_df_top = car_df_descending.head(50)
    bike_df_top = bike_df_descending.head(50)
    walk_df_top = walk_df_descending.head(50)

    dfs = [car_df_top, bike_df_top, walk_df_top]
    rider_types = ['CAR', 'BIKE', 'WALK']

    for itr in range(3):
        df = dfs[itr]
        rider_type = rider_types[itr]
        rider = riders_info[rider_type]
        used_seq = []

        for bundle1, bundle2 in combinations(df['Bundle'].tolist(), 2):
            merged_orders = set(bundle1.shop_seq + bundle2.shop_seq)
            if len(merged_orders) > 4:
                continue
            if merged_orders in used_seq:
                continue
            else:
                used_seq.append(merged_orders)

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
                new_df = pd.DataFrame(
                    {'Bundle': [optimal_bundle], 'Orders': [merged_orders], 'Reduced Cost': [reduced_cost],
                     'Rider': [rider_type]})
                bundles_df = pd.concat([bundles_df, new_df], ignore_index=True)

    return bundles_df

