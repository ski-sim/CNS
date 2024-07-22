from util import *
import pandas as pd
from itertools import permutations

def dynamicBundling(all_bundles, K, all_orders, riders_dict, dist_mat, timelimit=59):
    new_raw_dict = {'CAR Bundle':[],'BIKE Bundle':[],'WALK Bundle':[],'No':[],'CAR Reduced Cost':[],'BIKE Reduced Cost':[],'WALK Reduced Cost':[],'MAX Reduced Cost': [],'MAX Rider':[] }
    new_bundles = pd.DataFrame(new_raw_dict)

    for j in range(len(all_bundles)):
        for i in range(j):
            bundle1 = all_bundles[i]
            bundle2 = all_bundles[j]

            if len(bundle1.shop_seq) +  len(bundle2.shop_seq) > 4:
                continue


            cost1 = bundle1.cost
            cost2 = bundle2.cost
            reduced_cost = {'CAR':0,'BIKE':0,'WALK':0}
            optimal_bundle = {'CAR':None,'BIKE':None,'WALK':None}
            riders = ['CAR','BIKE','WALK']

            merged_orders = bundle1.shop_seq + bundle2.shop_seq
            total_volume = get_total_volume(all_orders, merged_orders)
            for rider_type in ['CAR','BIKE','WALK']:
                if total_volume > riders_dict[rider_type].capa :
                    riders.remove(rider_type)
            if len(riders) == 0:
                continue

            for shop_pem in permutations(merged_orders):
                for dlv_pem in permutations(merged_orders):
                    for rider_type in riders:
                        rider = riders_dict[rider_type]
                        feasibility_check = test_feasibility(all_orders, rider, shop_pem, dlv_pem)
                        if feasibility_check:
                            total_dist = get_total_distance(K, dist_mat, shop_pem, dlv_pem)
                            new_bundle = Bundle(all_orders, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
                            new_reduced_cost = cost1 + cost2 - new_bundle.cost
                            if new_reduced_cost > reduced_cost[rider_type]:
                                optimal_bundle[rider_type] = new_bundle
                                reduced_cost[rider_type] = new_reduced_cost

            if reduced_cost['CAR'] > 0 or reduced_cost['BIKE'] > 0 or reduced_cost['WALK'] > 0:
                max_reduced_cost = max(reduced_cost.values())
                max_rider = [k for k, v in reduced_cost.items() if v == max_reduced_cost]
                new_df = pd.DataFrame({'CAR Bundle':[optimal_bundle['CAR']],'BIKE Bundle':[optimal_bundle['BIKE']],'WALK Bundle':[optimal_bundle['WALK']],'No':[{i,j}],
                                       'CAR Reduced Cost':[reduced_cost['CAR']],'BIKE Reduced Cost':[reduced_cost['BIKE']],'WALK Reduced Cost':[reduced_cost['WALK']],
                                       'MAX Reduced Cost': [max_reduced_cost],'MAX Rider': max_rider })
                new_bundles = pd.concat([new_bundles,new_df],ignore_index=True)

    new_bundles_descending = new_bundles.sort_values(by='MAX Reduced Cost' ,ascending=False)
    return new_bundles_descending

def dynamicToList(dynamicBundles,old_bundles,riders_dict):
    bundle_list = []
    remaining = set(range(len(old_bundles)))
    for index,row in dynamicBundles.iterrows():
        number_set = row['No']
        if number_set <= remaining:
            new_rider_type = row['MAX Rider']
            new_rider = riders_dict[new_rider_type]
            old_riders_type = [old_bundles[i].rider.type for i in number_set]
            if new_rider_type in old_riders_type or new_rider.available_number > 0:
                for rider_type in old_riders_type:
                    riders_dict[rider_type].available_number += 1
                riders_dict[new_rider_type].available_number -= 1
                bundle_list.append(row[new_rider_type + ' Bundle'])
                remaining -= row['No']
    for itr in remaining:
        bundle_list.append(old_bundles[itr])
    return bundle_list

def test_feasibility(all_orders, rider, shop_seq, dlv_seq):
    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return False
    return True
