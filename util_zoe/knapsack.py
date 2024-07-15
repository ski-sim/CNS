import gurobipy as gp
from gurobipy import GRB

def bundleOptimization(df, K, riders_dict):
    model = gp.Model('Bundle Optimization')
    m = model.addVars(len(df),vtype=GRB.BINARY,name='m')
    bundles = df['Bundle'].tolist()
    rider_con = {'BIKE':0,'WALK':0, 'CAR':0}
    order_con = {}
    for i in range(K):
        order_con[i] = []

    model.setObjective(gp.quicksum(bundles[i].cost * m[i] for i in range(len(df))),GRB.MINIMIZE)

    for index, row in df.iterrows():
        orders = row['Orders']
        for i in orders:
            order_con[i].append(m[index])
        rider_type = row['Rider']
        rider_con[rider_type] += m[index]

    for i in range(K):
        model.addConstr(gp.quicksum(order_con[i]) == 1)
    for type in ['BIKE','WALK','CAR']:
        model.addConstr(rider_con[type] <= riders_dict[type].available_number)

    model.optimize()
    result = {}
    for itr in range(len(df)):
        result[itr] = m[itr].x

    indices = []
    for key, value in result.items():
        if value == 1:
            indices.append(key)

    return df.loc[indices]['Bundle'].tolist()