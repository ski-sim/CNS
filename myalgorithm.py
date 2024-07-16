from util import *
from util_LNS import *
from util_past.util_GA import *
import copy
def algorithm_regret(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    all_riders=[]
    temp_all_riders = my_all_riders.copy()
    for rider in temp_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    riders_num = {'CAR':car_rider.available_number,'BIKE':riders_dict['BIKE'].available_number,'WALK':riders_dict['WALK'].available_number}

    old_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        old_bundles.append(new_bundle)
        car_rider.available_number -= 1
    opt_bundles = old_bundles

    best_obj = sum((bundle.cost for bundle in opt_bundles)) / K
    print(f'Best obj = {best_obj}')

    # table 설정
    regret_table = create_regret_table(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)
    
    recent_bundle, updated_bundles, regret_table = update_bundle_and_regret_table(regret_table,old_bundles, riders_dict)
    
    regret_table = update_regret_table(all_orders, riders_dict,K, dist_mat, regret_table, updated_bundles, new_bundle)
    
    
    i=0
    while True:
        if len(regret_table)==0:
            break

        # table에서 최대 cost를 가지는 번들 선택, 해당 번들의 주문을 greedy_table, updated_bundles에서 제외 후 번들 추가
        new_bundle, updated_bundles, regret_table = update_bundle_and_regret_table(regret_table,updated_bundles, riders_dict)
        if new_bundle == None: # 최대 cost이지만 번들할당이 불가능할때
            continue

        # 묶인 번들에 대한 cost를 추가
        regret_table = update_regret_table(all_orders, riders_dict,K, dist_mat, regret_table, updated_bundles, new_bundle)
        if regret_table.empty: # 
            break

        if i%10==0:
            solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in updated_bundles
                ]

            checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, solution)
            print(i,'th current_solution',checked_solution['avg_cost'])  

        if time.time() - start_time > timelimit-0.5:
            break
        i+=1
    solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in updated_bundles
        ]
    print('end iteration')
    checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, solution)
    print('current_solution',checked_solution['avg_cost'])
    return recent_bundle, updated_bundles, regret_table



'''
algorithm_basic_greedy
'''
def algorithm_basic_greedy(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    
    all_riders=[]
    temp_all_riders = my_all_riders.copy()
    for rider in temp_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    riders_num = {'CAR':car_rider.available_number,'BIKE':riders_dict['BIKE'].available_number,'WALK':riders_dict['WALK'].available_number}

    old_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        old_bundles.append(new_bundle)
        car_rider.available_number -= 1
    opt_bundles = old_bundles

    best_obj = sum((bundle.cost for bundle in opt_bundles)) / K
    print(f'Best obj = {best_obj}')

    # table 설정
    greedy_table = create_greedy_table(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)

    i=0
    while True:
        if greedy_table.empty or len(greedy_table)==0: 
            break
        
        new_bundle, updated_bundles, greedy_table = update_bundle_and_greedy_table(greedy_table,old_bundles, riders_dict)
        if new_bundle == None: 
            continue
        else:
            greedy_table = update_greedy_table(all_orders, riders_dict,K, dist_mat, greedy_table, updated_bundles, new_bundle)
            if greedy_table.empty or len(greedy_table)==0: 
                break
            old_bundles = updated_bundles
        
        # print current solution
        if i%10==0:
            solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in updated_bundles
                ]
            checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, solution)
            print(i,'th current_solution',checked_solution['avg_cost'])  
        
        # time limit
        if time.time() - start_time > timelimit-0.5:
            break

        i+=1

    solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in old_bundles
        ]
    

    print('end iteration')
    checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, solution)
    print('current_solution',checked_solution['avg_cost'])
    return solution #, opt_bundles_group
    
'''
algorithm_basic_greedy + relatedness
'''
def algorithm_basic_greedy_v2(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    
    all_riders=[]
    temp_all_riders = my_all_riders.copy()
    for rider in temp_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    riders_dict = {}
    for r in all_riders:
        riders_dict[r.type] = r
    car_rider = riders_dict['CAR']

    riders_num = {'CAR':car_rider.available_number,'BIKE':riders_dict['BIKE'].available_number,'WALK':riders_dict['WALK'].available_number}

    old_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        old_bundles.append(new_bundle)
        car_rider.available_number -= 1
    opt_bundles = old_bundles

    best_obj = sum((bundle.cost for bundle in opt_bundles)) / K
    print(f'Best obj = {best_obj}')

    # table 설정
    greedy_table = create_greedy_table(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)

    i=0
    while True:
        if greedy_table.empty or len(greedy_table)==0: 
            break
        
        new_bundle, updated_bundles, greedy_table = update_bundle_and_greedy_table(greedy_table,old_bundles, riders_dict)
        if new_bundle == None: 
            continue
        else:
            greedy_table = update_greedy_table(all_orders, riders_dict,K, dist_mat, greedy_table, updated_bundles, new_bundle)
            if greedy_table.empty or len(greedy_table)==0: 
                break
            old_bundles = updated_bundles
        
        # print current solution
        if i%10==0:
            greedy_solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in updated_bundles
                ]
            checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, greedy_solution)
            print(i,'th current_solution',checked_solution['avg_cost'])  
        
        # time limit
        if time.time() - start_time > timelimit-0.5:
            break

        i+=1

    greedy_best_solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in old_bundles
        ]
    checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, greedy_best_solution)
    
    bike_count = sum(1 for item in greedy_best_solution if item[0] == "BIKE")
    walk_count = sum(1 for item in greedy_best_solution if item[0] == "WALK")
    car_count = sum(1 for item in greedy_best_solution if item[0] == "CAR")

    temp_all_riders = copy.deepcopy(my_all_riders)
    riders_dict = {}
    for r in temp_all_riders:
        riders_dict[r.type] = r
    riders_dict['BIKE'].available_number-= bike_count
    riders_dict['WALK'].available_number-= walk_count
    riders_dict['CAR'].available_number-= car_count


    alpha, beta, gamma, p = 0.3, 0.4, 0.5, 3 # hyper parameter
    relatedness = calculate_relatedness(all_orders, dist_mat,K,alpha, beta, gamma)

    q=5 # hyper parameter
    best_obj = checked_solution['avg_cost']
    best_bundles = old_bundles
    print(f'performance before local search : {best_obj}')

    iteration = True
    while iteration:
        
        lns_solution = None
        my_rider_dict = copy.deepcopy(riders_dict)
        D = calcuate_D(relatedness,K,p,q)
        remain_order = copy.deepcopy(D)
        current_bundles = copy.deepcopy(old_bundles)
        remove_bundles = [] 
        remove_shop_seqs = []  

        for bundle in current_bundles:
            intersection = list(set(D) & set(bundle.shop_seq))
            if len(intersection) != 0: # 교집합이 있을때,
                D = list(set(D) - set(intersection)) # 해당 부분 D에서 제거
                new_shop_seq = list(set(bundle.shop_seq) - set(intersection)) #기존 리스트에 교집합 부분 제거거
                if len(new_shop_seq) != 0:
                    remove_shop_seqs.append(new_shop_seq) # 추후에 다시 삽입
                remove_bundles.append(bundle) # 기존 번들은 삭제
        
        for bundle in remove_bundles:
            rider_type = bundle.rider.type
            my_rider_dict[rider_type].available_number += 1
            current_bundles.remove(bundle)


        for new_shop_seq in remove_shop_seqs:
            new_bundle,my_rider_dict = check_crossover_feasibility(new_shop_seq, my_rider_dict, all_orders, dist_mat, K)
            if new_bundle == None: # best rider로 바뀌면서 기존 rider가 부족할 수 있음
                for shop_seq in new_shop_seq:
                    shop_seq_as_list = [shop_seq]  
                    new_bundle,my_rider_dict = check_crossover_feasibility(shop_seq_as_list, my_rider_dict, all_orders, dist_mat, K)
                    current_bundles.append(new_bundle)
                    rider_type = new_bundle.rider.type
                    my_rider_dict[rider_type].available_number -= 1 
            else:
                current_bundles.append(new_bundle)
                rider_type = new_bundle.rider.type
                my_rider_dict[rider_type].available_number -= 1 # 추가
        
        updated_bundles = merge_remain(current_bundles, remain_order, my_rider_dict, all_orders, car_rider, dist_mat, K)

        lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in updated_bundles
                    ]


        checked_solution = solution_check(K, all_orders, my_all_riders, dist_mat, lns_solution)
        current_cost = checked_solution['avg_cost']

        if best_obj > current_cost:
            best_obj = current_cost
            best_bundles = updated_bundles
            old_bundles = updated_bundles
            riders_dict = my_rider_dict
            print(f'local search cost:',current_cost)
            
        if time.time() - start_time > timelimit-0.1:
            print('break')
            iteration=False

    best_lns_solution = [
                [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
                for bundle in best_bundles
                    ]
    print('best')
    return best_lns_solution

