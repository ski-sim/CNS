from util import *
import pandas as pd
from itertools import permutations
from itertools import combinations
import copy

def binaryMerging(bundles_input, order_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60):
    riders_info = copy.deepcopy(riders_dict)
    bundles_dict = {'Bundle':[],'Orders':[],'Reduced Cost':[],'Rider':[] }

    car_df = bundles_input[ bundles_input['Rider']=='CAR' ]
    bike_df = bundles_input[bundles_input['Rider'] == 'BIKE']
    walk_df = bundles_input[bundles_input['Rider'] == 'WALK']
    dfs = [car_df,bike_df,walk_df]
    rider_types = ['CAR','BIKE','WALK']

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
            if total_volume > riders_info[rider_type].capa :
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

def initialMerging(K, all_orders, riders_dict, dist_mat, timelimit=60):
    riders_info = copy.deepcopy(riders_dict)
    bundles_dict = {'Bundle':[],'Orders':[],'Reduced Cost':[],'Rider':[] }

    order_bundles = []

    for order in all_orders:
        total_dist = get_total_distance(K, dist_mat, [order.id], [order.id] )
        car_bundle = Bundle(all_orders, riders_info['CAR'], [order.id], [order.id], order.volume, total_dist)
        order_bundles.append(car_bundle)
        bundles_dict['Bundle'].append(car_bundle)
        bundles_dict['Orders'].append({order.id})
        bundles_dict['Reduced Cost'].append(0)
        bundles_dict['Rider'].append('CAR')

        for rider_type in ['BIKE', 'WALK']:
            if order.volume > riders_info[rider_type].capa:
                continue
            rider = riders_info[rider_type]
            if test_feasibility(all_orders, rider, [order.id], [order.id]):
                new_bundle = Bundle(all_orders, rider, [order.id], [order.id], order.volume, total_dist)
                bundles_dict['Bundle'].append(new_bundle)
                bundles_dict['Orders'].append({order.id})
                bundles_dict['Reduced Cost'].append(car_bundle.cost - new_bundle.cost)
                bundles_dict['Rider'].append(rider_type)

    bundles_df = pd.DataFrame(bundles_dict)

    return bundles_df, order_bundles

#Gurobi를 쓰는 관계로 더이상 사용하지 않음
def bundle_df_ToList(bundles_df,all_orders,riders_dict):
    riders_info = copy.deepcopy(riders_dict)
    bundle_list = []
    bundles_df_descending = bundles_df.sort_values(by='Reduced Cost', ascending=False)

    remaining = set(range(len(all_orders)))

    for index,row in bundles_df_descending.iterrows():
        number_set = set(row['Orders'])
        if number_set <= remaining:
            new_rider_type = row['Rider']
            new_rider = riders_info[new_rider_type]
            if new_rider.available_number > 0:
                bundle_list.append(row['Bundle'])
                new_rider.available_number -= 1
                remaining -= number_set
    return bundle_list



def test_feasibility(all_orders, rider, shop_seq, dlv_seq):
    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return False
    return True

#RiderOpt는 사용하지 않고, Gurobi로 대체함
def riderOpt(new_bundles,riders_dict,all_orders):
    new_raw_dict = {'Bundle':[],'Orders':[],'Reduced Cost':[],'Rider':[] }
    new_pd = pd.DataFrame(new_raw_dict)
    final_bundles = []
    remaining = list(range(len(new_bundles)))

    for itr in range(len(new_bundles)):
        bundle = new_bundles[itr]
        riders = ['CAR','BIKE','WALK']
        riders.remove(bundle.rider.type)
        for rider_type in riders:
            rider = riders_dict[rider_type]
            if rider.capa >= bundle.total_volume:
                feasibility_check = test_feasibility(all_orders, rider, bundle.shop_seq, bundle.dlv_seq)
                if feasibility_check:
                    new_bundle = Bundle(all_orders, rider, list(bundle.shop_seq), list(bundle.dlv_seq), bundle.total_volume, bundle.total_dist)
                    reduced_cost = bundle.cost - new_bundle.cost
                    if reduced_cost > 0 :
                        df = pd.DataFrame({'Bundle': [new_bundle], 'No': [itr], 'Reduced Cost': [reduced_cost], 'Rider': [rider_type]})
                        new_pd = pd.concat([new_pd, df], ignore_index=True)
    new_pd_descending = new_pd.sort_values(by='Reduced Cost', ascending=False)

    for index,row in new_pd_descending.iterrows():
        number = row['Orders']
        if number in remaining:
            new_rider_type = row['Rider']
            new_rider = riders_dict[new_rider_type]
            old_rider_type = new_bundles[int(number)].rider.type
            old_rider = riders_dict[old_rider_type]
            if new_rider.available_number > 0:
                old_rider.available_number += 1
                new_rider.available_number -= 1
                final_bundles.append(row['Bundle'])
                remaining.remove(number)
    for itr in remaining:
        final_bundles.append(new_bundles[itr])

    return final_bundles


