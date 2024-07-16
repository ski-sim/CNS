from util import *
import pandas as pd
from itertools import permutations
def add_regret(greedy_table,target_order):
    if target_order in greedy_table['Primary Order'].unique():
        filtered_df = greedy_table[greedy_table['Primary Order'] == target_order]
        max_cost_row = filtered_df.loc[filtered_df['MAX Reduced Cost'].idxmax()]
        max_cost = max_cost_row['MAX Reduced Cost']
        second_max_cost = filtered_df.loc[filtered_df['MAX Reduced Cost'] != max_cost, 'MAX Reduced Cost'].max()
        third_max_cost = filtered_df.loc[(filtered_df['MAX Reduced Cost'] != max_cost)&
                                         (filtered_df['MAX Reduced Cost'] != second_max_cost), 'MAX Reduced Cost'].max()
        
        # 두 값의 차이 계산 (두 번째로 큰 값이 없으면 최대 값을 그대로 사용)
        if pd.isna(second_max_cost):
            cost_difference = max_cost
        else:
            cost_difference = max_cost - second_max_cost

        if pd.isna(second_max_cost):
            cost_difference = max_cost
        else:
            cost_difference += max_cost - third_max_cost

        greedy_table.loc[greedy_table['MAX Reduced Cost'] == max_cost, 'regret_cost'] = cost_difference
        new_bundles_descending = greedy_table.sort_values(by='regret_cost', ascending=False)

        return new_bundles_descending
    else:
        return greedy_table
    
def add_initial_regret(greedy_table):
    greedy_table['regret_cost'] = 0.0
    unique_primary_orders = greedy_table['Primary Order'].unique()

    for primary_order_value in unique_primary_orders:
        # Primary Order가 현재 고유 값과 같은 모든 행 필터링
        filtered_df = greedy_table[greedy_table['Primary Order'] == primary_order_value]
        
        # MAX Reduced Cost 열에서 최대 값과 두 번째로 큰 값 찾기
        max_cost_row = filtered_df.loc[filtered_df['MAX Reduced Cost'].idxmax()]
        max_cost = max_cost_row['MAX Reduced Cost']
        second_max_cost = filtered_df.loc[filtered_df['MAX Reduced Cost'] != max_cost, 'MAX Reduced Cost'].max()
        third_max_cost = filtered_df.loc[(filtered_df['MAX Reduced Cost'] != max_cost)&
                                         (filtered_df['MAX Reduced Cost'] != second_max_cost), 'MAX Reduced Cost'].max()
        
        # 두 값의 차이 계산 (두 번째로 큰 값이 없으면 최대 값을 그대로 사용)
        if pd.isna(second_max_cost):
            cost_difference = max_cost
        else:
            cost_difference = max_cost - second_max_cost

        if pd.isna(second_max_cost):
            cost_difference = max_cost
        else:
            cost_difference += max_cost - third_max_cost
        
        # 최대 값 행의 regret_cost 열을 차이 값으로 업데이트
        greedy_table.loc[greedy_table['MAX Reduced Cost'] == max_cost, 'regret_cost'] = cost_difference
        
    new_bundles_descending = greedy_table.sort_values(by='regret_cost', ascending=False)

    return new_bundles_descending

def create_regret_table(all_bundles, K, all_orders, riders_dict, dist_mat, timelimit=59):
    # 기존의 데이터 저장을 위한 딕셔너리 초기화
    result_dict = {
        'Primary Order': [],
        'Secondary Order': [],
        'CAR Bundle': [],
        'BIKE Bundle': [],
        'WALK Bundle': [],
        'CAR Reduced Cost': [],
        'BIKE Reduced Cost': [],
        'WALK Reduced Cost': [],
        'MAX Reduced Cost': [],
        'MAX Rider': []
    }

    # 기존의 all_bundles 리스트를 기반으로 연산 수행
    for j in range(len(all_bundles)):
        for i in range(j):
            bundle1 = all_bundles[i]
            bundle2 = all_bundles[j]

            if len(bundle1.shop_seq) + len(bundle2.shop_seq) > 4:
                continue

            cost1 = bundle1.cost
            cost2 = bundle2.cost
            reduced_cost = {'CAR': 0, 'BIKE': 0, 'WALK': 0}
            optimal_bundle = {'CAR': None, 'BIKE': None, 'WALK': None}
            riders = ['CAR', 'BIKE', 'WALK']

            merged_orders = bundle1.shop_seq + bundle2.shop_seq
            total_volume = get_total_volume(all_orders, merged_orders)
            for rider_type in ['CAR', 'BIKE', 'WALK']:
                if total_volume > riders_dict[rider_type].capa:
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

                # 결과 딕셔너리에 값 추가
                result_dict['Primary Order'].append(bundle1)
                result_dict['Secondary Order'].append(bundle2)
                result_dict['CAR Bundle'].append(optimal_bundle['CAR'])
                result_dict['BIKE Bundle'].append(optimal_bundle['BIKE'])
                result_dict['WALK Bundle'].append(optimal_bundle['WALK'])
                result_dict['CAR Reduced Cost'].append(reduced_cost['CAR'])
                result_dict['BIKE Reduced Cost'].append(reduced_cost['BIKE'])
                result_dict['WALK Reduced Cost'].append(reduced_cost['WALK'])
                result_dict['MAX Reduced Cost'].append(max_reduced_cost)
                result_dict['MAX Rider'].append(max_rider[0])

    # 결과 딕셔너리를 데이터프레임으로 변환
    greedy_table = pd.DataFrame(result_dict)
    
    # return greedy_table
    regret_table = add_initial_regret(greedy_table)

    return regret_table


def update_bundle_and_regret_table(regret_table,old_bundle,riders_dict):
    updated_bundle = old_bundle.copy()
    first_row = regret_table.iloc[0,:]
    new_rider_type = first_row['MAX Rider']
    new_rider = riders_dict[new_rider_type]
    old_riders_type = [first_row['Order1'].rider.type,first_row['Order2'].rider.type]


    if new_rider_type in old_riders_type or new_rider.available_number > 0:
        for rider_type in old_riders_type:
            riders_dict[rider_type].available_number += 1
        riders_dict[new_rider_type].available_number -= 1
        
        updated_bundle.remove(first_row['Order1'])
        updated_bundle.remove(first_row['Order2'])
        updated_bundle.append(first_row[new_rider_type + ' Bundle'])

        # Order1, Order2에 대한 내용 삭제
        regret_table= regret_table[~regret_table['Order1'].isin([first_row['Order1'], first_row['Order2']])]       
        regret_table= regret_table[~regret_table['Order2'].isin([first_row['Order1'], first_row['Order2']])]

    else:
        regret_table = regret_table[~((regret_table['Order1'] == first_row['Order1']) & (regret_table['Order2'] == first_row['Order2']))]
        regret_table = regret_table[~((regret_table['Order2'] == first_row['Order1']) & (regret_table['Order1'] == first_row['Order2']))]
        
        regret_table = add_regret(regret_table, first_row['Order1'])
        return  None, updated_bundle, regret_table
    
    return first_row[new_rider_type + ' Bundle'], updated_bundle, regret_table


def update_regret_table(all_orders, riders_dict,K, dist_mat, regret_table, updated_bundles, new_bundle):
    updated_regret_table = regret_table.copy()
    bundle1 = new_bundle

    for j in range(len(updated_bundles)):
        bundle2 = updated_bundles[j]

        if bundle1 ==  bundle2:
            continue

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
            new_df = pd.DataFrame({'Primary Order':[bundle1],'Secondary Order':[bundle2],
                                    'CAR Bundle':[optimal_bundle['CAR']],'BIKE Bundle':[optimal_bundle['BIKE']],'WALK Bundle':[optimal_bundle['WALK']],
                                    'CAR Reduced Cost':[reduced_cost['CAR']],'BIKE Reduced Cost':[reduced_cost['BIKE']],'WALK Reduced Cost':[reduced_cost['WALK']],
                                    'MAX Reduced Cost': [max_reduced_cost],'MAX Rider': max_rider })
            updated_regret_table = pd.concat([updated_regret_table,new_df],ignore_index=True)

    
    updated_regret_table = add_regret(updated_regret_table, new_bundle) 

    new_bundles_descending = updated_regret_table.sort_values(by='regret_cost', ascending=False)
   
    return new_bundles_descending


################################################################################################################

def create_greedy_table(all_bundles, K, all_orders, riders_dict, dist_mat, timelimit=59):
    # 기존의 데이터 저장을 위한 딕셔너리 초기화
    result_dict = {
        'Order1': [],
        'Order2': [],
        'CAR Bundle': [],
        'BIKE Bundle': [],
        'WALK Bundle': [],
        'CAR Reduced Cost': [],
        'BIKE Reduced Cost': [],
        'WALK Reduced Cost': [],
        'MAX Reduced Cost': [],
        'MAX Rider': []
    }

    # 기존의 all_bundles 리스트를 기반으로 연산 수행
    for j in range(len(all_bundles)):
        for i in range(j):
            bundle1 = all_bundles[i]
            bundle2 = all_bundles[j]

            if len(bundle1.shop_seq) + len(bundle2.shop_seq) > 4:
                continue

            cost1 = bundle1.cost
            cost2 = bundle2.cost
            reduced_cost = {'CAR': 0, 'BIKE': 0, 'WALK': 0}
            optimal_bundle = {'CAR': None, 'BIKE': None, 'WALK': None}
            riders = ['CAR', 'BIKE', 'WALK']

            merged_orders = bundle1.shop_seq + bundle2.shop_seq
            total_volume = get_total_volume(all_orders, merged_orders)
            for rider_type in ['CAR', 'BIKE', 'WALK']:
                if total_volume > riders_dict[rider_type].capa:
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

                # 결과 딕셔너리에 값 추가
                result_dict['Order1'].append(bundle1)
                result_dict['Order2'].append(bundle2)
                result_dict['CAR Bundle'].append(optimal_bundle['CAR'])
                result_dict['BIKE Bundle'].append(optimal_bundle['BIKE'])
                result_dict['WALK Bundle'].append(optimal_bundle['WALK'])
                result_dict['CAR Reduced Cost'].append(reduced_cost['CAR'])
                result_dict['BIKE Reduced Cost'].append(reduced_cost['BIKE'])
                result_dict['WALK Reduced Cost'].append(reduced_cost['WALK'])
                result_dict['MAX Reduced Cost'].append(max_reduced_cost)
                result_dict['MAX Rider'].append(max_rider[0])

    # 결과 딕셔너리를 데이터프레임으로 변환
    new_bundles = pd.DataFrame(result_dict)

    # MAX Reduced Cost를 기준으로 내림차순 정렬
    new_bundles_descending = new_bundles.sort_values(by='MAX Reduced Cost', ascending=False)

    return new_bundles_descending

# 
def update_greedy_table(all_orders, riders_dict,K, dist_mat, greedy_table, updated_bundles, new_bundle):
    new_greedy_table = greedy_table.copy()
    bundle1 = new_bundle

    for j in range(len(updated_bundles)):
        bundle2 = updated_bundles[j]
        if bundle1 ==  bundle2:
            continue
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
            new_df = pd.DataFrame({'Order1':[bundle1],'Order2':[bundle2],
                                    'CAR Bundle':[optimal_bundle['CAR']],'BIKE Bundle':[optimal_bundle['BIKE']],'WALK Bundle':[optimal_bundle['WALK']],
                                    'CAR Reduced Cost':[reduced_cost['CAR']],'BIKE Reduced Cost':[reduced_cost['BIKE']],'WALK Reduced Cost':[reduced_cost['WALK']],
                                    'MAX Reduced Cost': [max_reduced_cost],'MAX Rider': max_rider })
            new_greedy_table = pd.concat([new_greedy_table,new_df],ignore_index=True)



    new_greedy_table = new_greedy_table.sort_values(by='MAX Reduced Cost' ,ascending=False)
   
    return new_greedy_table

# table에서 최대 cost를 가지는 번들 선택, 해당 번들의 주문을 greedy_table, updated_bundles에서 제외 후 번들 추가
def update_bundle_and_greedy_table(greedy_table,old_bundle,riders_dict,all_orders,K,dist_mat):
    
    updated_bundles = old_bundle.copy()
    first_row = greedy_table.iloc[0,:]
    new_rider_type = first_row['MAX Rider']
    new_rider = riders_dict[new_rider_type]
    old_riders_types = [first_row['Order1'].rider.type,first_row['Order2'].rider.type]

    if new_rider_type in old_riders_types or new_rider.available_number > 0:
        for old_rider_type in old_riders_types:
            riders_dict[old_rider_type].available_number += 1

        riders_dict[new_rider_type].available_number -= 1

        new_bundle = first_row[new_rider_type + ' Bundle']
        updated_bundles.remove(first_row['Order1'])
        updated_bundles.remove(first_row['Order2'])
        updated_bundles.append(first_row[new_rider_type + ' Bundle'])

        greedy_table= greedy_table[~greedy_table['Order1'].isin([first_row['Order1'], first_row['Order2']])]
        greedy_table= greedy_table[~greedy_table['Order2'].isin([first_row['Order1'], first_row['Order2']])]
        greedy_table = update_greedy_table(all_orders, riders_dict,K, dist_mat, greedy_table, updated_bundles, new_bundle)
        
        return updated_bundles, greedy_table
    else:
        greedy_table = greedy_table[~((greedy_table['Order1'] == first_row['Order1']) & (greedy_table['Order2'] == first_row['Order2']))]
        greedy_table = greedy_table[~((greedy_table['Order2'] == first_row['Order1']) & (greedy_table['Order1'] == first_row['Order2']))]
        
        return  updated_bundles, greedy_table
    
    


def test_feasibility(all_orders, rider, shop_seq, dlv_seq):
    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return False
    return True


def get_normalizing_factors(all_orders, dist_mat):
    
    dist_max = np.max(dist_mat)
    dist_min = np.min(dist_mat)

    volumes = [order.volume for order in all_orders]
    v_max = max(volumes)
    v_min = min(volumes)

    ready_times = [order.ready_time for order in all_orders]
    r_max = max(ready_times)
    r_min = min(ready_times)

    deadlines = [order.deadline for order in all_orders]
    d_max = max(deadlines)
    d_min = min(deadlines)

    return dist_max, dist_min, v_max, v_min, r_max, r_min, d_max, d_min


def calculate_relatedness(all_orders, dist_mat,K,alpha,beta,gamma):
    
    N = len(all_orders)
    relatedness = np.zeros((N, N))

    dist_max, dist_min, v_max, v_min, r_max, r_min, d_max, d_min = get_normalizing_factors(all_orders, dist_mat)

    for i in range(N):
        for j in range(N):
            relatedness[i, j] = (
                alpha*((dist_mat[i, j] - dist_min)/(dist_max-dist_min) + (dist_mat[i + K, j + K]-dist_min)/(dist_max-dist_min))
                + beta*((all_orders[i].volume-v_min)/(v_max-v_min) - (all_orders[j].volume-v_min)/(v_max-v_min))
                + gamma*(abs((all_orders[i].ready_time-r_min)/(r_max-r_min) - (all_orders[j].ready_time-r_min)/(r_max-r_min)) \
                        + abs((all_orders[i].deadline-d_min)/(d_max-d_min) - (all_orders[j].deadline-d_min)/(d_max-d_min)))
            )

    return relatedness

def calcuate_D(relatedness,K,p,q):
    D = []
    D.append(np.random.randint(0, K)) 
    while len(D)<q:
        r = random.choice(D)
        L = relatedness[r]
        sorted_indices = np.argsort(L)
        y = np.random.uniform(0, 1)**p
        y = int(np.floor(y*len(L))) 
        candidate = sorted_indices[y]
        
        if candidate not in D:  # 중복된 요소가 없는 경우에만 추가
            D.append(candidate)
    return D