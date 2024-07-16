from util import *

def create_population(all_bundles,K,dist_mat, all_orders):
    population =[]
    for i in range(100):
        my_all_bundles = all_bundles.copy()
        iter=0
        best_obj = 9999
        while iter < 4:

            bundle1, bundle2 = select_two_bundles(my_all_bundles)
            new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)

            if new_bundle is not None:
                my_all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1
                
                my_all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                my_all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                # cur_obj = sum((bundle.cost for bundle in my_all_bundles)) / K
                # if cur_obj < best_obj:
                #     best_obj = cur_obj
                #     print(f'Best obj = {best_obj}')
                iter += 1

            else:
                iter += 0
        
        population.append(my_all_bundles)
    print('create initial population')
    return population

def findBestSolution(population,K):
    idxMaximum = -1
    best_obj = 99999
    best_bundle=None
    for itr in range(len(population)):
        cur_obj = sum((bundle.cost for bundle in population[itr])) / K
      
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_bundle = population[itr]
            
    return best_bundle,best_obj

def fitness(my_all_bundles,K):
    cur_obj = sum((bundle.cost for bundle in my_all_bundles)) / K
    return 1000/cur_obj

def calculate_culmulative_prob(population,K):
        
    rankFitness = {}
    originalFitness = {}
    maxUtility = -999999
    minUtility = 999999
    for itr in range(len(population)):
        originalFitness[itr] = fitness(population[itr],K)
        if maxUtility < originalFitness[itr]:
            maxUtility = originalFitness[itr]
        if minUtility > originalFitness[itr]:
            minUtility = originalFitness[itr]

    # 적합도 값을 정렬 (내림차순)
    sorted_fitness = sorted(originalFitness.items(), key=lambda item: item[1], reverse=False)

    # 적합도 값의 총합 계산
    total_fitness = sum(fitness for index, fitness in sorted_fitness)

    # 정규화된 확률 계산
    probabilities = [(index, fitness / total_fitness) for index, fitness in sorted_fitness]

    # 누적 확률 계산
    cumulative_probabilities = []
    cumulative_sum = 0
    for index, probability in probabilities:
        cumulative_sum += probability
        cumulative_probabilities.append((index, cumulative_sum))

    return cumulative_probabilities
# 확률적으로 부모 선택
def select_parent_index(cumulative_probabilities):
    r = random.random()  # 0과 1 사이의 난수 생성
    for index, cumulative_probability in cumulative_probabilities:
        if r <= cumulative_probability:
            return index

# 부모 선택 함수
def select_parents(population,K):
    cumulative_probabilities = calculate_culmulative_prob(population,K)

    parent1 = select_parent_index(cumulative_probabilities)
    parent2 = select_parent_index(cumulative_probabilities)

    # 부모가 같으면 다시 선택
    while parent1 == parent2:
        parent2 = select_parent_index(cumulative_probabilities)
 
    return population[parent1], population[parent2]

def substitutePopulation(population, children,K):
    population.sort(key=lambda ind: fitness(ind, K))

    for itr in range(len(children)):
        population[itr] = children[itr]

    return population

def calculateCost(instance,ALL_RIDERS,K,DIST):
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in instance
    ]
    total_cost =0
    for bundle_info in solution:
        # Get rider object
        rider = None
        rider_type = bundle_info[0]
        for r in ALL_RIDERS:
            if r.type == rider_type:
                rider = r
        dist_mat = rider.T        
        dist = get_total_distance(K, DIST, bundle_info[1], bundle_info[2])
        cost = rider.calculate_cost(dist)


        total_cost += cost

    avg_cost = total_cost/K
    return avg_cost

def crossover_shop_seq(list1, list2):
    merged_list = []
    max_length = max(len(list1), len(list2))

    for i in range(max_length):
        if i < len(list1) and i >= len(list2):
            merged_list.append(list1[i])
        elif i < len(list2) and i >= len(list1):
            merged_list.append(list2[i])
        else:
            if random.choice([True, False]):
                merged_list.append(list1[i])
            else:
                merged_list.append(list2[i])

    return list(set(merged_list))
def test_feasibility(all_orders, rider, shop_seq, dlv_seq):
    pickup_times, dlv_times = get_pd_times(all_orders, rider, shop_seq, dlv_seq)
    for k, dlv_time in dlv_times.items():
        if dlv_time > all_orders[k].deadline:
            return False
    return True

def check_crossover_feasibility(new_shop_seq,riders_dict,ALL_ORDERS,DIST,K):

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
        
        #riders_dict[max_rider[0]].available_number-=1
        return optimal_bundle[max_rider[0]],riders_dict
    else:
        return None ,riders_dict   


    
def merge_remain_eachother(remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K):
    
    remain_bundles = []

    for ord in remain_order:
        riders_dict['CAR'].available_number -=1
        new_bundle = Bundle(ALL_ORDERS, car_rider, [ALL_ORDERS[ord].id], [ALL_ORDERS[ord].id], ALL_ORDERS[ord].volume, DIST[ALL_ORDERS[ord].id, ALL_ORDERS[ord].id+K])
        remain_bundles.append(new_bundle)

    temp_bundles = remain_bundles.copy()
    i_values = temp_bundles
    j_values = temp_bundles

    skip_condition = []
    i = 0
    while i < len(i_values):
        bundle1 = i_values[i]
        if bundle1 in skip_condition:
            i += 1
            continue
        j = 0
        while j < len(j_values):
            if i ==j:
                j += 1
                continue
            bundle2 = j_values[j]
            if bundle2 in skip_condition:
                j += 1
                continue
            if len(bundle1.shop_seq) +  len(bundle2.shop_seq) > 4:
                j += 1
                continue

            riders = ['CAR','BIKE','WALK']
            merged_orders = bundle1.shop_seq + bundle2.shop_seq
            new_bundle,riders_dict = check_crossover_feasibility(merged_orders,riders_dict,ALL_ORDERS,DIST,K)
            if new_bundle == None:
                j += 1
                break
            else:
                skip_condition.append(bundle1)
                skip_condition.append(bundle2)
                riders_dict['CAR'].available_number +=2
                remain_bundles.append(new_bundle)
                j += 1
                break

        i += 1
   
    for i in skip_condition:
        remain_bundles.remove(i)
    return remain_bundles  


def greedy_check_crossover_feasibility(new_shop_seq,riders_dict,ALL_ORDERS,DIST,K):

    reduced_cost = {'CAR':99999,'BIKE':99999,'WALK':99999}
    optimal_bundle = {'CAR':None,'BIKE':None,'WALK':None}
    riders = ['CAR','BIKE','WALK']
    total_volume = get_total_volume(ALL_ORDERS, new_shop_seq)
    for rider_type in ['CAR','BIKE','WALK']:
        if total_volume > riders_dict[rider_type].capa :
            riders.remove(rider_type)
    if len(riders) == 0:
        return None,riders_dict

    indices = [1, 2, 3, 4,5,6]
    pickup_start_index, min_ready_time_order = min(
        ((i, ALL_ORDERS[i]) for i in indices), 
        key=lambda x: x[1].ready_time
    )
 
    delivery_end_index, max_deadline_time_order = max(
        ((i, ALL_ORDERS[i]) for i in indices), 
        key=lambda x: x[1].deadline
    )
    
    shop_start = new_shop_seq[pickup_start_index]
    dlv_end = new_shop_seq[delivery_end_index]
    
    all_shop_pem = new_shop_seq.copy()
    all_dlv_pem = new_shop_seq.copy()
    
    all_shop_pem.pop(pickup_start_index)
    all_dlv_pem.pop(delivery_end_index)
  
    for shop_pem in permutations(all_shop_pem):
        shop_pem = (shop_start,) + shop_pem
      
        for dlv_pem in permutations(all_dlv_pem):
            dlv_pem =  dlv_pem + (dlv_end,)
            
            for rider_type in riders:
                rider = riders_dict[rider_type]
                if rider.available_number<=0:
                    continue
                feasibility_check = test_feasibility(ALL_ORDERS, rider, shop_pem, dlv_pem)
                if feasibility_check:
                    total_dist = get_total_distance(K, DIST, shop_pem, dlv_pem)

                    new_bundle = Bundle(ALL_ORDERS, rider, list(shop_pem), list(dlv_pem), total_volume, total_dist)
    #                 riders_dict[rider_type].available_number -=1
    #                 return new_bundle,riders_dict
    # return None, riders_dict
                    # if len(new_shop_seq)>3:
                    #     riders_dict[rider_type].available_number -=1
                    #     return new_bundle,riders_dict
                    
                    if new_bundle.cost < reduced_cost[rider_type]:
                        optimal_bundle[rider_type] = new_bundle
                        reduced_cost[rider_type] = new_bundle.cost 
                       

    if reduced_cost['CAR'] < 99999 or reduced_cost['BIKE']  < 99999 or reduced_cost['WALK']  < 99999:
        max_reduced_cost = min(reduced_cost.values())
        max_rider = [k for k, v in reduced_cost.items() if v == max_reduced_cost]
        riders_dict[max_rider[0]].available_number-=1
        return optimal_bundle[max_rider[0]],riders_dict
    else:
        return None ,riders_dict   

def crossover(parent1, parent2,K,riders_dict,ALL_ORDERS,DIST,car_rider):
    # 길이가 짧은 것을 기준

    if len(parent1) > len(parent2):
        p1, p2 = parent2, parent1
    else:
        p1, p2 = parent1, parent2

    p2_copy = p2[:]
    child_bundle = []
    total_shop_seq = []
    zero_count = 0
    new_bundle = None
    for p1_bundle in p1:
        for p2_bundle in p2_copy:
            if not set(p1_bundle.shop_seq).isdisjoint(set(p2_bundle.shop_seq)):
                new_shop_seq = crossover_shop_seq(p1_bundle.shop_seq,p2_bundle.shop_seq)
                if 0 in new_shop_seq and zero_count>=1:
                    break
                if not any(set(new_shop_seq) & set(total_shop_seq)):
                    if 0 in new_shop_seq :
                        zero_count+=1
                    new_bundle,riders_dict = check_crossover_feasibility(new_shop_seq,riders_dict,ALL_ORDERS,DIST,K)
                    if new_bundle != None:
                        total_shop_seq+=new_shop_seq
            
                        child_bundle.append(new_bundle)
                        break
                else:
                    continue
                
    total_order = list(range(K))
    remain_order = list(set(total_order) - set(total_shop_seq))
    
    # 방법 2
    child_bundle = merge_remain(child_bundle, remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K)
    # 방법 1
    # remain_bundles = merge_remain_eachother(remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K)
    # child_bundle += remain_bundles
    return child_bundle,total_shop_seq,remain_order, riders_dict


def merge_remain_one_step(child_bundle, remain_bundle, riders_dict, ALL_ORDERS, DIST, K):
    best_bundle = None
    remove_bundle = None
    best_cost = float('inf')
    new_bundle = None

    for child in child_bundle:
        new_order = remain_bundle.shop_seq + child.shop_seq
        if len(new_order) >= 4:
            continue
        else:# check_crossover 함수 안에서 rider 개수를 조정하면 가능한 조합 모두 반영되어서 혼동됨
            new_bundle, riders_dict = check_crossover_feasibility(new_order, riders_dict, ALL_ORDERS, DIST, K)
        
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

def merge_remain(current_bundles, remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K):
    my_current_bundles = current_bundles.copy()
    for ord in remain_order:
        riders_dict['CAR'].available_number -=1
        remain_bundle = Bundle(ALL_ORDERS, car_rider, [ALL_ORDERS[ord].id], [ALL_ORDERS[ord].id], ALL_ORDERS[ord].volume, DIST[ALL_ORDERS[ord].id, ALL_ORDERS[ord].id+K])
        update_bundles, success = merge_remain_one_step(my_current_bundles, remain_bundle, riders_dict, ALL_ORDERS, DIST, K)
        my_current_bundles = update_bundles
    return my_current_bundles

#####################################################################################3
import pandas as pd
def process_bundle(row, remain_order, riders_dict, ALL_ORDERS, DIST, K):
    new_order = remain_order + row['shop_seq']
    if len(new_order) > 4:
        return None, riders_dict, None  # Skip to the next iteration
    new_bundle_cost = None
    new_bundle, riders_dict = check_crossover_feasibility(new_order, riders_dict, ALL_ORDERS, DIST, K)
    if new_bundle != None :
        new_bundle_cost = new_bundle.cost
    return new_bundle, riders_dict, new_bundle_cost


def merge_remain_one_step2(child_bundle,remain_bundle, riders_dict, ALL_ORDERS, DIST, K):
    
    child_bundle_df = pd.DataFrame({
    'shop_seq': [bundle.shop_seq for bundle in child_bundle]
    })

    # Apply the function to each row
    results = child_bundle_df.apply(
        lambda row: process_bundle(row, remain_bundle.shop_seq, riders_dict, ALL_ORDERS, DIST, K), axis=1
    )

    results_df = pd.DataFrame(results.tolist(), columns=['new_bundle', 'riders_dict', 'new_bundle_cost'])
    results_df['original_index'] = results_df.index
    valid_bundles =None
    valid_bundles = results_df[results_df['new_bundle'].notnull()]
    if not valid_bundles.empty:
        max_cost_index = valid_bundles['new_bundle_cost'].idxmax()
        max_cost_bundle = valid_bundles.loc[max_cost_index, 'new_bundle']
        original_index_to_remove = valid_bundles.loc[max_cost_index, 'original_index']
    
        # print("New bundle with the highest cost:", max_cost_bundle)
        child_bundle.append(max_cost_bundle)
        child_bundle.remove(child_bundle[original_index_to_remove])
        return child_bundle, valid_bundles
    else:
        child_bundle.append(remain_bundle)
        return child_bundle, valid_bundles
    
def merge_remain2(child_bundle, remain_order,riders_dict,ALL_ORDERS,car_rider,DIST,K):

    for ord in remain_order:
        riders_dict['CAR'].available_number -=1
        remain_bundle = Bundle(ALL_ORDERS, car_rider, [ALL_ORDERS[ord].id], [ALL_ORDERS[ord].id], ALL_ORDERS[ord].volume, DIST[ALL_ORDERS[ord].id, ALL_ORDERS[ord].id+K])
    
        child_bundle, valid_bundles = merge_remain_one_step2(child_bundle, remain_bundle, riders_dict, ALL_ORDERS, DIST, K)
    

    return child_bundle