from util import *
from util_LNS import *
from util_past.geneticAlgorithm import *
from pastalgorithm import *
import copy


def algorithm_regret(K, all_orders, all_riders, dist_mat, timelimit=60, lns=False):
    start_time = time.time()
    
    my_all_riders = copy.deepcopy(all_riders)
    for r in my_all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    riders_dict = {}
    for r in my_all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    old_bundles = []
    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        old_bundles.append(new_bundle)
        car_rider.available_number -= 1
    

    best_obj = sum((bundle.cost for bundle in old_bundles)) / K
    print(f'Best obj = {best_obj}')

    #------------- Custom algorithm code starts from here --------------#
    greedy_table = create_regret_table(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)

    i=0
    while True:
        updated_bundles, greedy_table = update_bundle_and_regret_table(greedy_table,old_bundles, riders_dict,all_orders,K,dist_mat)
        old_bundles = updated_bundles
        print_avg_cost(i, old_bundles,K, all_orders, all_riders, dist_mat)

        if greedy_table.empty or len(greedy_table)==0: 
            break
        if time.time() - start_time > timelimit-0.5:
            break
        i+=1

    solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in old_bundles
        ]
    
    checked_solution = solution_check(K, all_orders, all_riders, dist_mat, solution)
    print('best_solution',checked_solution['avg_cost'])

    if lns == True:
        return updated_bundles, checked_solution, riders_dict
    else:
        return solution 
    

'''
algorithm_basic_greedy
'''
def algorithm_basic_greedy(K, all_orders, all_riders, dist_mat, timelimit=60, lns=False):
    start_time = time.time()
    
    my_all_riders = copy.deepcopy(all_riders)
    for r in my_all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    riders_dict = {}
    for r in my_all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    old_bundles = []
    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        old_bundles.append(new_bundle)
        car_rider.available_number -= 1
    

    best_obj = sum((bundle.cost for bundle in old_bundles)) / K
    print(f'Best obj = {best_obj}')

    #------------- Custom algorithm code starts from here --------------#
    greedy_table = create_greedy_table(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)

    i=0
    while True:
        updated_bundles, greedy_table = update_bundle_and_greedy_table(greedy_table,old_bundles, riders_dict,all_orders,K,dist_mat)
        old_bundles = updated_bundles
        print_avg_cost(i, old_bundles,K, all_orders, all_riders, dist_mat)

        if greedy_table.empty or len(greedy_table)==0: 
            break
        if time.time() - start_time > timelimit-0.5:
            break
        i+=1

    solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in old_bundles
        ]
    
    checked_solution = solution_check(K, all_orders, all_riders, dist_mat, solution)
    print('best_solution',checked_solution['avg_cost'])

    if lns == True:
        return updated_bundles, checked_solution, riders_dict
    else:
        return solution 
    
'''
algorithm_basic_greedy + relatedness
'''
def algorithm_LNS(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    
    all_riders=[]
    temp_all_riders = my_all_riders.copy()
    for rider in temp_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    #------------- Custom algorithm code starts from here --------------#

    initial_bundles, initial_solution, riders_dict = algorithm_basic_greedy(K, all_orders, my_all_riders, dist_mat, timelimit=60, lns=True)
    # initial_bundles, initial_solution, riders_dict = algorithm_regret(K, all_orders, my_all_riders, dist_mat, timelimit=60, lns=True)
    
    best_obj = initial_solution['avg_cost']
    best_bundles = initial_bundles
    print(f'performance before local search : {best_obj}')

    alpha = 0.3 # 0~1
    beta = 0.4 # 0~1
    gamma = 0.5 # 0~1
    p = 3
    q = 10 # 100개당 5개 
    w = 0.01 # 0~1
    T = w * best_obj / np.log(0.5)
    C = 0.7
    relatedness = calculate_relatedness(all_orders, dist_mat, K, alpha, beta, gamma)

    acc = 1 
    iter = True
    while iter:
        if acc%2==0:
            method ='random'
        else:
            method = 'shaw'
        current_bundles, removed_orders, current_rider_dict = lns_removal(best_bundles, relatedness, all_orders, dist_mat, riders_dict, K, p, q,method)
        updated_bundles = lns_insertion(current_bundles, removed_orders, current_rider_dict, all_orders, car_rider, dist_mat, K)
        # updated_bundles = lns_insertion2(current_bundles, removed_orders, current_rider_dict, all_orders, car_rider, dist_mat, K)

        lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in updated_bundles
                    ]
        checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, lns_solution)
        current_cost = checked_solution['avg_cost']

        if simple_acceptance(current_cost, best_obj):
                best_obj = current_cost
                best_bundles = updated_bundles
                current_bundles = updated_bundles
                riders_dict = current_rider_dict
                print(f'simple local search cost:',current_cost)


        if time.time() - start_time > timelimit-0.1:
            print('break')
            iter = False
        
        T = T*C
        acc+=1
    best_lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in best_bundles
                    ]
    return best_lns_solution

    
'''
algorithm_basic_greedy + adaptive selection
'''
def algorithm_ALNS(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    
    all_riders=[]
    temp_all_riders = my_all_riders.copy()
    for rider in temp_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    #------------- Custom algorithm code starts from here --------------#

    initial_bundles, initial_solution, riders_dict = algorithm_basic_greedy(K, all_orders, my_all_riders, dist_mat, timelimit=60, lns=True) 
    best_obj = initial_solution['avg_cost']
    best_bundles = initial_bundles
    print(f'performance before local search : {best_obj}')

    alpha = 0.3 # 0~1
    beta = 0.4 # 0~1
    gamma = 0.5 # 0~1
    p = 3
    q = 10 # 100개당 5개 
    w = 0.01 # 0~1
    T = w * best_obj / np.log(0.5)
    C = 0.7
    relatedness = calculate_relatedness(all_orders, dist_mat, K, alpha, beta, gamma)

    acc = 1 
    iter = True
    
    weights = [0.5,0.5] 
    heuristics = ['random','shaw']
    scores = [0,0]
    counts = [0,0]
    r = 0.7
    acc = 0

    bundles_table = set()
    while iter:

        acc+=1
        if acc % 100==0:
            for i in range(len(weights)):
                weights[i] = weights[i]*(1-r)+r*scores[i]/counts[i]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            scores = [0,0]
            counts = [0,0]
        method = roulette_selection(heuristics,weights)
        idx = heuristics.index(method)
        current_bundles, removed_orders, current_rider_dict = lns_removal(best_bundles, relatedness, all_orders, dist_mat, riders_dict, K, p, q,method)
        updated_bundles = lns_insertion(current_bundles, removed_orders, current_rider_dict, all_orders, car_rider, dist_mat, K)
        # updated_bundles = lns_insertion2(current_bundles, removed_orders, current_rider_dict, all_orders, car_rider, dist_mat, K)

        lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in updated_bundles
                    ]
        checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, lns_solution)
        current_cost = checked_solution['avg_cost']

        if simple_acceptance(current_cost, best_obj):
            best_obj = current_cost
            best_bundles = updated_bundles
            current_bundles = updated_bundles
            riders_dict = current_rider_dict
            print(f'simple local search cost:',current_cost)
            scores[idx] += 33

        if time.time() - start_time > timelimit-0.1:
            print('break')
            iter = False

        updated_bundles_tuple = tuple(updated_bundles)
        if updated_bundles_tuple not in bundles_table:
            bundles_table.add(updated_bundles_tuple)
            scores[idx] += 13
            
        counts[idx] += 1
            
    best_lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in best_bundles
                    ]
    return best_lns_solution