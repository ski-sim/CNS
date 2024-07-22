from util import *
from util_past.clustering import *
from util_past.dynamicProgramming import *
from util_past.geneticAlgorithm import *


def algorithm_GA(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    timelimit -=1
    start_time = time.time()
    all_riders=[]
    for rider in my_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
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
    
    all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1
    
    # create population  
    print('creating population')  
    population = create_population(all_bundles,K,dist_mat, all_orders)
    print('finish creating population')
    best_bundle, best_obj = findBestSolution(population,K)
    print(f'Best obj = {best_obj}')

    genetic_population = population.copy()


    iter=0

    while iter <2000:
        offsprings = []
        for itr in range(70):    
            #
            fixed_all_riders=[]
            for rider in all_riders:
                info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
                fixed_all_riders.append(Rider(info))
            for r in fixed_all_riders:
                r.T = np.round(dist_mat/r.speed + r.service_time)

            riders_dict={}
            for r in fixed_all_riders:
                temp = r
                riders_dict[r.type] = temp
            
            # select parent
            p1, p2 = select_parents(genetic_population,K)
            
            child,_,_,_ = crossover(p1, p2,K,riders_dict,all_orders,dist_mat,car_rider)

            offsprings.append(child)
            if time.time() - start_time > timelimit:
                break
            
        print(time.time() - start_time)
        genetic_population = substitutePopulation(genetic_population, offsprings,K)
        mutation_probability = iter / 1000
        
        best,best_obj = findBestSolution(genetic_population,K)

        print(f'{iter}th iteration, avg_cost :{best_obj}')

        solution = [
            [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
            for bundle in best
        ]
        checked_solution = solution_check(K, all_orders, all_riders, dist_mat, solution)

        iter+=1
        if time.time() - start_time > timelimit:
            break
    return genetic_population
    return solution


'''
merge 하였을 때, cost가 가장 감소하는 순으로 merge
rider 별로 cost도 비교해서 최적 rider로 merge
'''
def algorithm_DP(K, all_orders, my_all_riders, dist_mat, timelimit=60):
    all_riders=[]
    for rider in my_all_riders:
        info = [rider.type,rider.speed,rider.capa,rider.var_cost,rider.fixed_cost, rider.service_time, rider.available_number]
        all_riders.append(Rider(info))
    
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    solution = []

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
    
    #------------- Custom algorithm code starts from here --------------#   
    opt_bundles_group =[]
    for i in range(1):
        car_rider.available_number += len(old_bundles)
        dynamic_bundles = dynamicBundling(old_bundles, K, all_orders, riders_dict, dist_mat, timelimit=60)
        dynamic_bundles_list = dynamicToList(dynamic_bundles,old_bundles, riders_dict)
        new_bundles = dynamic_bundles_list[:]
        old_bundles = dynamic_bundles_list[:]
        car_rider.available_number = riders_num['CAR'] - len(new_bundles) - riders_num['BIKE'] + riders_dict['BIKE'].available_number - riders_num['WALK'] + riders_dict['WALK'].available_number

        cur_obj = sum((bundle.cost for bundle in new_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')
            opt_bundles = new_bundles
            opt_bundles_group.append(opt_bundles)
        if time.time() - start_time > timelimit:
            break
      

    solution = [

        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in opt_bundles
    ]

    return solution,dynamic_bundles
    

'''
주문 수 별로 clustering 개수를 달리함. 주문수에 비례하여 증가
클러스터링으로 분리된 작은 문제를 greedy하게 해결
'''
def algorithm_clustering_greedy(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    car_rider = None
    for r in all_riders:
        if r.type == 'CAR':
            car_rider = r

    all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')

    
    # 1. 클러스터링 실행    
    all_orders_df =  k_means_clustering(K//100+1, all_orders,42)

    # 2. 클러스터링 결과로 전체 주문을 분리
    sub_bundles=[]
    for i in range(all_orders_df['cluster'].nunique()):
        cluster_index= list(all_orders_df[all_orders_df['cluster']==i]['ID'])
        sub_bundles.append([all_bundles[i] for i in cluster_index])

    
    # 3. 자동차를 기준으로 먼저 greedy하게 merge
    # 590개 까지
    for iter_total in range(len(sub_bundles)):
        iter_cluster = iter_total%len(sub_bundles)
        iter = 0
        max_merge_iter = 1000

        # 조합 계산 후 거리기준으로 정렬
        combination_2 = []
        sub_bundles_id = [i.shop_seq[0] for i in sub_bundles[iter_cluster]]
        for i in [i.shop_seq[0] for i in sub_bundles[iter_cluster]]:
            for j in [i.shop_seq[0] for i in sub_bundles[iter_cluster]]:
                if j in sub_bundles_id and i!=j:

                    dist_i = dist_mat[i][i+K]
                    dist_j = dist_mat[j][i+K]

                    combination_2.append([(i, j),np.mean([dist_i,dist_j])])

            sub_bundles_id.pop(0)

        combination_2 = sorted(combination_2, key=lambda x: x[1])

        print(f'all possible combination :{len(combination_2)}')
        while iter < len(combination_2):
            # if iter> 2400//(K//100+1): # 최신버전 590
            #     break
            if iter>  590:
                break
            bundle1 = None
            bundle2 = None
        
            for bundle in all_bundles:
                if combination_2[iter][0][0] in bundle.shop_seq and bundle1==None:
                    bundle1 = bundle

                if combination_2[iter][0][1] in bundle.shop_seq and bundle2==None:  
                    bundle2 = bundle
        
                if bundle1 !=None and bundle2 !=None:
                    break
                
                
            if bundle1 !=bundle2:
                new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
            else:
                new_bundle = None

            if new_bundle is not None:
                all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1
                
                all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')
                iter += 1
            else:
                iter += 1

            if time.time() - start_time > timelimit:
                break

        if time.time() - start_time > timelimit:
            break
        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')
        iter_total+=1
        print(iter_total)
        if iter_total==len(sub_bundles):
            print('vehicle change')
    

    for iter_total in range(len(sub_bundles)):

        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(all_orders, dist_mat, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                if time.time() - start_time > timelimit:
                    break


        
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#



    return solution

   
'''
1. 두개로 구성된 가능한 묶음 조합을 계산
2. 각 조합마다 주문 쌍의 픽업-배달 거리 평균 계산
3. 픽업-배달 거리 평균이 짧은 순으로 묶음
4. 초기에 정렬된 주문 쌍 순서로 계속 묶음 시도함
'''
def algorithm_greedy(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    car_rider = None
    for r in all_riders:
        if r.type == 'CAR':
            car_rider = r

    all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')


    #################### 2 묶음으로 가능한 조합 계산 ####################
    comb2 = []
    for i in range(K):
        for j in range(i + 1, K):
            
            dist_i = dist_mat[i][i+K]
            dist_j = dist_mat[j][i+K]
            
            comb2.append([(i, j),np.mean([dist_i,dist_j])])

    comb2 = sorted(comb2, key=lambda x: x[1])
    comb2 = [item for item in comb2 if item[1] <= 2000]
    

    Train=True
    while Train:

        iter = 0
        max_merge_iter = 1000
        
        while iter < max_merge_iter:

            bundle1 = None
            bundle2 = None
            if iter<len(comb2):
                for bundle in all_bundles:
                    if comb2[iter][0][0] in bundle.shop_seq and bundle1==None:
                        bundle1 = bundle
 
                    if comb2[iter][0][1] in bundle.shop_seq and bundle2==None:  
                        bundle2 = bundle
            
                    if bundle1 !=None and bundle2 !=None:
                        # if bundle1 == bundle2:
                        #     bundle1, bundle2 = select_two_bundles(all_bundles)
                        #     break
                        break
                
                
            else:
                print('random')
                
                break
                
            if bundle1 !=bundle2:
                new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)
            else:
                new_bundle = None

            if new_bundle is not None:
                all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1
                
                all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')
                iter += 1
            else:
                iter += 1

            if time.time() - start_time > timelimit:
                break

        if time.time() - start_time > timelimit:
            break


        
        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(all_orders, dist_mat, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                if time.time() - start_time > timelimit:
                    break

        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')


    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#



    return solution

def algorithm_baseline(K, all_orders, all_riders, dist_mat, timelimit=60):

    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat/r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    #------------- Custom algorithm code starts from here --------------#

    car_rider = None
    for r in all_riders:
        if r.type == 'CAR':
            car_rider = r

    all_bundles = []

    for ord in all_orders:
        new_bundle = Bundle(all_orders, car_rider, [ord.id], [ord.id], ord.volume, dist_mat[ord.id, ord.id+K])
        all_bundles.append(new_bundle)
        car_rider.available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')


    # Very stupid random merge algorithm
    while True:

        iter = 0
        max_merge_iter = 1000
        
        while iter < max_merge_iter:

            bundle1, bundle2 = select_two_bundles(all_bundles)
            new_bundle = try_merging_bundles(K, dist_mat, all_orders, bundle1, bundle2)

            if new_bundle is not None:
                all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1
                
                all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')

            else:
                iter += 1

            if time.time() - start_time > timelimit:
                break

        if time.time() - start_time > timelimit:
            break


        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(all_orders, dist_mat, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                if time.time() - start_time > timelimit:
                    break


        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')


    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    #------------- End of custom algorithm code--------------#



    return solution
    