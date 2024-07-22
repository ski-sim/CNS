import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

def to_radians(degree):
    return degree * (3.141592653589793 / 180)

def sin(x):
    return (4 * x * (180 - x)) / (40500 - x * (180 - x))

def cos(x):
    x = to_radians(x)
    return (sin(90 - x))

def sqrt(x):
    return x**0.5

def atan2(y, x):
    if x > 0:
        return atan(y / x)
    elif x < 0 and y >= 0:
        return atan(y / x) + 3.141592653589793
    elif x < 0 and y < 0:
        return atan(y / x) - 3.141592653589793
    elif x == 0 and y > 0:
        return 3.141592653589793 / 2
    elif x == 0 and y < 0:
        return -3.141592653589793 / 2
    else:
        return 0

def atan(x):
    return x - (x**3)/3 + (x**5)/5 - (x**7)/7 + (x**9)/9


def haversine_distance(row):
    lat1, lon1, lat2, lon2 = row['shop_lat'], row['shop_lon'], row['dlv_lat'], row['dlv_lon']
    
    R = 6371.0

    # 좌표를 라디안으로 변환
    lat1_rad = to_radians(lat1)
    lon1_rad = to_radians(lon1)
    lat2_rad = to_radians(lat2)
    lon2_rad = to_radians(lon2)

    # 차이를 계산
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # 하버사인 공식 적용
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # 거리 계산
    distance = R * c

    return distance

def k_means_clustering(n_clusters, ALL_ORDERS,random_state):
    order_list = []
    for order in ALL_ORDERS:
        order_list.append({
            'ID' : order.id,
            'ready_time' : order.ready_time,
            'deadline' : order.deadline,
            'shop_lat' : order.shop_lat,
            'shop_lon' : order.shop_lon,
            'dlv_lat' : order.dlv_lat,
            'dlv_lon' : order.dlv_lon,
        })

    order_df = pd.DataFrame(order_list)
    order_df['mean'] = order_df.apply(haversine_distance, axis=1)
    
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    order_df['cluster'] = kmeans.fit_predict(order_df[['ready_time','mean']])


    return order_df

def visualize_k_means(order_df,k):
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))

    # 클러스터별로 색상 구분
    colors = cm.rainbow(np.linspace(0, 1, k))
    for cluster in range(k):
        clustered_data = order_df[order_df['cluster'] == cluster]
        plt.scatter(clustered_data['shop_lat'], clustered_data['shop_lon'], c=colors[cluster], label=f'Cluster {cluster}')

    for cluster in range(k):
        clustered_data = order_df[order_df['cluster'] == cluster]
        plt.scatter(clustered_data['dlv_lat'], clustered_data['dlv_lon'], c=colors[cluster])

    # 매장과 배송지를 잇는 선을 그립니다.
    for cluster in range(k):
        clustered_data = order_df[order_df['cluster'] == cluster]
        for sx, sy, dx, dy in zip(clustered_data['shop_lat'], clustered_data['shop_lon'], clustered_data['dlv_lat'], clustered_data['dlv_lon']):

            plt.plot([sx, dx], [sy, dy], 'black', linestyle='-', linewidth=0.5)  # 회색 선으로 매장과 배송지를 연결


    # 그래프 라벨링
    plt.title('K-means Clustering of Orders')
    plt.xlabel('Mean')
    plt.ylabel('Radius')
    plt.legend()
    plt.grid(True)
    plt.show()

def k_means_interval(n_clusters, ALL_ORDERS,random_state):
    order_list = []
    for order in ALL_ORDERS:
        order_list.append({
            'ID' : order.id,
            'mean' : (order.ready_time+order.deadline)/2, 
            'radius' : (-order.ready_time+order.deadline)/2,
            'shop_lat' : order.shop_lat,
            'shop_lon' : order.shop_lon,
            'dlv_lat' : order.dlv_lat,
            'dlv_lon' : order.dlv_lon,
        })

    order_df = pd.DataFrame(order_list)
    order_df['mean'] = (order_df['mean']-order_df['mean'].min())/(order_df['mean'].max()-order_df['mean'].min())
    order_df['radius'] = (order_df['radius']-order_df['radius'].min())/(order_df['radius'].max()-order_df['radius'].min())
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    order_df['cluster'] = kmeans.fit_predict(order_df[['mean', 'radius']])

    return order_df