from util import *
import pandas as pd
import copy
from itertools import permutations
def add_regret(greedy_table,target_order):
    if target_order in greedy_table['Order1'].unique():
        filtered_df = greedy_table[greedy_table['Order1'] == target_order]
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
    unique_primary_orders = greedy_table['Order1'].unique()

    for primary_order_value in unique_primary_orders:
        # Order1가 현재 고유 값과 같은 모든 행 필터링
        filtered_df = greedy_table[greedy_table['Order1'] == primary_order_value]
        
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
    greedy_table = pd.DataFrame(result_dict)
    
    # return greedy_table
    regret_table = add_initial_regret(greedy_table)

    return regret_table


def update_bundle_and_regret_table(regret_table,old_bundle,riders_dict,all_orders,K,dist_mat):
    updated_bundles = old_bundle.copy()
    first_row = regret_table.iloc[0,:]
    new_rider_type = first_row['MAX Rider']
    new_rider = riders_dict[new_rider_type]
    old_riders_type = [first_row['Order1'].rider.type,first_row['Order2'].rider.type]


    if new_rider_type in old_riders_type or new_rider.available_number > 0:
        for rider_type in old_riders_type:
            riders_dict[rider_type].available_number += 1
        riders_dict[new_rider_type].available_number -= 1
        
        new_bundle = first_row[new_rider_type + ' Bundle']
        updated_bundles.remove(first_row['Order1'])
        updated_bundles.remove(first_row['Order2'])
        updated_bundles.append(new_bundle)

        # Order1, Order2에 대한 내용 삭제
        regret_table= regret_table[~regret_table['Order1'].isin([first_row['Order1'], first_row['Order2']])]       
        regret_table= regret_table[~regret_table['Order2'].isin([first_row['Order1'], first_row['Order2']])]
        regret_table = update_regret_table(all_orders, riders_dict,K, dist_mat, regret_table, updated_bundles, new_bundle)
        return  updated_bundles, regret_table
    
    else:
        regret_table = regret_table[~((regret_table['Order1'] == first_row['Order1']) & (regret_table['Order2'] == first_row['Order2']))]
        regret_table = regret_table[~((regret_table['Order2'] == first_row['Order1']) & (regret_table['Order1'] == first_row['Order2']))]
        
        regret_table = add_regret(regret_table, first_row['Order1'])
        return  updated_bundles, regret_table
    



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
            new_df = pd.DataFrame({'Order1':[bundle1],'Order2':[bundle2],
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
        updated_bundles.append(new_bundle)

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

def lns_removal(old_bundles, relatedness, all_orders, dist_mat, riders_dict, K, p, q,method='shaw'):
    if method == 'shaw':
        D = shaw_removal(relatedness,K,p,q)
    elif method == 'worse':
        D = worse_removal(old_bundles, all_orders, dist_mat, K, p, q)
    elif method == 'random':
        D = random_removal(K,q)
    
    
    current_rider_dict = copy.deepcopy(riders_dict)
    removed_orders = copy.deepcopy(D)
    current_bundles = copy.deepcopy(old_bundles) 
    removed_shop_seqs = []  

    # D 와 겹치는 bundle 제거 
    for bundle in current_bundles[:]:
        intersection = list(set(D) & set(bundle.shop_seq))
        if len(intersection) != 0: # 교집합이 있을때,
            D = list(set(D) - set(intersection)) # 해당 부분 D에서 제거
            new_shop_seq = list(set(bundle.shop_seq) - set(intersection)) #기존 리스트에 교집합 부분 제거거
            if len(new_shop_seq) != 0:
                removed_shop_seqs.append(new_shop_seq) # 추후에 다시 삽입

            rider_type = bundle.rider.type
            current_rider_dict[rider_type].available_number += 1
            current_bundles.remove(bundle) # 기존 번들은 삭제
    
    # D가 포함되지 않은 번들 생성
    for new_shop_seq in removed_shop_seqs:
            new_bundle,current_rider_dict = check_insertion_feasibility(new_shop_seq, current_rider_dict, all_orders, dist_mat, K)
            if new_bundle == None: # best rider로 바뀌면서 기존 rider가 부족할 수 있음
                for shop_seq in new_shop_seq:
                    shop_seq_as_list = [shop_seq]  
                    new_bundle,current_rider_dict = check_insertion_feasibility(shop_seq_as_list, current_rider_dict, all_orders, dist_mat, K)
                    current_bundles.append(new_bundle)
                    rider_type = new_bundle.rider.type
                    current_rider_dict[rider_type].available_number -= 1 
            else:
                current_bundles.append(new_bundle)
                rider_type = new_bundle.rider.type
                current_rider_dict[rider_type].available_number -= 1 # 추가

    return current_bundles, removed_orders, current_rider_dict

def shaw_removal(relatedness,K,p,q):
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

def random_removal(K,q):
    D = []
    while len(D)<q:
        candidate = candidate = random.randint(0, K-1)
        if candidate not in D:  # 중복된 요소가 없는 경우에만 추가
            D.append(candidate)
    return D

# 현재 솔루션에서 bundle 하나씩 방문
# 번들 내 원소 마다 하나씩 지우고 그 해를 찾고
# 감소된 cost를 찾고
# 그걸로 sorting
def worse_removal(old_bundles, all_orders, dist_mat, K, p, q):
    L = list(range(0,100))
    for bundle in old_bundles:

        for order in bundle.shop_seq:
            
            new_shop_seq = bundle.shop_seq[:].remove(order)
            new_dlv_seq = bundle.dlv_seq[:].remove(order)
            if  new_shop_seq != None:
                total_volume = get_total_volume(all_orders, new_shop_seq)
                total_dist = get_total_distance(K, dist_mat, new_shop_seq, new_dlv_seq)
                new_bundle = Bundle(all_orders, bundle.rider, new_shop_seq, new_dlv_seq, total_volume, total_dist)
                reduced_cost = bundle.cost - new_bundle.cost
                L[order] = reduced_cost 
            else:
                L[order] = 0 

    D = []
    while len(L)<q:
        sorted_indices = np.argsort(L)[::-1]
        y = np.random.uniform(0, 1)**p
        y = int(np.floor(y*len(L))) 
        candidate = sorted_indices[y]
        
        if candidate not in D:  # 중복된 요소가 없는 경우에만 추가
            D.append(candidate)
    return D



def check_remove_feasibility(new_shop_seq,rider, ALL_ORDERS,DIST,K):
    best_bundle = None
    best_cost = float('inf')
    total_volume = get_total_volume(ALL_ORDERS, new_shop_seq)

    for shop_pem in permutations(new_shop_seq):
        for dlv_pem in permutations(new_shop_seq):
            feasibility_check = test_feasibility(ALL_ORDERS, rider, shop_pem, dlv_pem)
            if feasibility_check:
                total_dist = get_total_distance(K, DIST, shop_pem, dlv_pem)
                new_bundle = Bundle(ALL_ORDERS, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
                if best_cost > new_bundle.cost:
                    best_bundle = new_bundle
                    best_cost = new_bundle.cost  
    return best_bundle

def check_insertion_feasibility(new_shop_seq,riders_dict,ALL_ORDERS,DIST,K):

    reduced_cost = {'CAR':99999,'BIKE':99999,'WALK':99999}
    optimal_bundle = {'CAR':None,'BIKE':None,'WALK':None}
    riders = ['CAR','BIKE','WALK']

    total_volume = get_total_volume(ALL_ORDERS, new_shop_seq)
    for rider_type in ['CAR','BIKE','WALK']:
        if total_volume > riders_dict[rider_type].capa :
            riders.remove(rider_type)
    if len(riders) == 0:
        return None,riders_dict

    for shop_pem in permutations(new_shop_seq):
        for dlv_pem in permutations(new_shop_seq):
            for rider_type in riders:
                rider = riders_dict[rider_type]
                if rider.available_number<=0:
                    continue

                feasibility_check = test_feasibility(ALL_ORDERS, rider, shop_pem, dlv_pem)
                if feasibility_check:
                    total_dist = get_total_distance(K, DIST, shop_pem, dlv_pem)
                    new_bundle = Bundle(ALL_ORDERS, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
                    
                    if new_bundle.cost < reduced_cost[rider_type]:
                        optimal_bundle[rider_type] = new_bundle
                        reduced_cost[rider_type] = new_bundle.cost 
                       

    if reduced_cost['CAR'] < 99999 or reduced_cost['BIKE']  < 99999 or reduced_cost['WALK']  < 99999: 
        max_reduced_cost = min(reduced_cost.values())
        max_rider = [k for k, v in reduced_cost.items() if v == max_reduced_cost]
        return optimal_bundle[max_rider[0]],riders_dict
    else:
        return None ,riders_dict   


def basic_greedy_insertion(child_bundle, remain_bundle, riders_dict, ALL_ORDERS, DIST, K):
    best_bundle = None
    remove_bundle = None
    best_cost = float('inf')
    new_bundle = None

    for child in child_bundle:
        new_order = remain_bundle.shop_seq + child.shop_seq
        if len(new_order) >= 4:
            continue
        else:# check_crossover 함수 안에서 rider 개수를 조정하면 가능한 조합 모두 반영되어서 혼동됨
            new_bundle, riders_dict = check_insertion_feasibility(new_order, riders_dict, ALL_ORDERS, DIST, K)
        
            if new_bundle is not None:
                if new_bundle.cost <= best_cost:
                    best_cost = new_bundle.cost
                    best_bundle = new_bundle
                    remove_bundle = child

    if best_bundle is not None and remove_bundle is not None:
        child_bundle.append(best_bundle)
        child_bundle.remove(remove_bundle)
        origin_rider_type = remove_bundle.rider.type
        new_rider_type = best_bundle.rider.type
        riders_dict[new_rider_type].available_number -= 1
        riders_dict[origin_rider_type].available_number += 1
        riders_dict['CAR'].available_number += 1
    
    else:
        child_bundle.append(remain_bundle)

    return child_bundle, new_bundle

def lns_insertion(current_bundles, remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K):
    my_current_bundles = current_bundles.copy()
    for ord in remain_order:
        riders_dict['CAR'].available_number -=1
        remain_bundle = Bundle(ALL_ORDERS, car_rider, [ALL_ORDERS[ord].id], [ALL_ORDERS[ord].id], ALL_ORDERS[ord].volume, DIST[ALL_ORDERS[ord].id, ALL_ORDERS[ord].id+K])
        update_bundles, success = basic_greedy_insertion(my_current_bundles, remain_bundle, riders_dict, ALL_ORDERS, DIST, K)
        my_current_bundles = update_bundles
    return my_current_bundles

def lns_insertion2(current_bundles, remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K):
    my_current_bundles = current_bundles.copy()

    result_dict = {'current_bundle':[], 'new_bundle':[], 'remain_bundle':[], 'reduced_cost':[], 'current_rider':[], 'new_rider':[]}
    for ord in remain_order:
        remain_bundle = Bundle(ALL_ORDERS, car_rider, [ALL_ORDERS[ord].id], [ALL_ORDERS[ord].id], ALL_ORDERS[ord].volume, DIST[ALL_ORDERS[ord].id, ALL_ORDERS[ord].id+K])
        best_bundle = None
        remove_bundle = None
        best_cost = float('inf')
        new_bundle = None

        for child in my_current_bundles:
            new_order = remain_bundle.shop_seq + child.shop_seq
            if len(new_order) >= 4:
                continue
            else:# check_crossover 함수 안에서 rider 개수를 조정하면 가능한 조합 모두 반영되어서 혼동됨
                new_bundle, riders_dict = check_insertion_feasibility(new_order, riders_dict, ALL_ORDERS, DIST, K)
                if new_bundle is not None:
                    result_dict['current_bundle'].append(child)
                    result_dict['new_bundle'].append(new_bundle)
                    result_dict['remain_bundle'].append(ord)
                    result_dict['reduced_cost'].append(child.cost + remain_bundle.cost - new_bundle.cost)
                    result_dict['current_rider'].append(child.rider.type)
                    result_dict['new_rider'].append(new_bundle.rider.type)
        
        result_dict['current_bundle'].append(remain_bundle)
        result_dict['new_bundle'].append(remain_bundle)
        result_dict['remain_bundle'].append(ord)
        result_dict['reduced_cost'].append(0)
        result_dict['current_rider'].append(child.rider.type)
        result_dict['new_rider'].append(remain_bundle.rider.type)
    result_df = pd.DataFrame(result_dict)
    result_df = result_df.sort_values(by='reduced_cost' ,ascending=False)

    remaining = []
    for index,row in result_df.iterrows():
  
        if row['remain_bundle'] in remaining or row['current_bundle'] in remaining:
            continue
        # rider 가능 여부 확인
        new_rider_type = row['new_rider']
        new_rider = riders_dict[new_rider_type]
        old_riders_type = row['current_rider']
        if new_rider_type == old_riders_type or new_rider.available_number > 0:
            
            riders_dict[new_rider_type].available_number -= 1

            if row['current_bundle'] != row['new_bundle']:
                riders_dict[old_riders_type].available_number += 1
                my_current_bundles.remove(row['current_bundle'])

            my_current_bundles.append(row['new_bundle'])
            remaining.append(row['current_bundle'])
            remaining.append(row['remain_bundle'])

    return my_current_bundles

def print_avg_cost(i, current_bundles,K, all_orders, all_riders, dist_mat):
    if i%50==0:
        solution = [
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in current_bundles
            ]
        checked_solution = solution_check(K, all_orders, all_riders, dist_mat, solution)
        print(i,'th current_solution',checked_solution['avg_cost'])  
        return None
    else:
        return None
    
def simple_acceptance(current_cost, best_obj):
    return best_obj > current_cost

def SA_acceptance(current_cost, best_obj, T):
    accept_prob = np.exp(-(current_cost - best_obj) / T)
    random_value = np.random.rand()

    return random_value < accept_prob

def roulette_selection(heuristics,weights):
    roulette = []
    prob = 0
    for w in weights:
        prob+=w
        roulette.append(prob)
    
    dart = random.random()  # 다트 던지기
    selected_heuristic = None
    for idx in range(len(roulette)):
        if dart < roulette[idx]:
            selected_heuristic = heuristics[idx]
            break

    return selected_heuristic
