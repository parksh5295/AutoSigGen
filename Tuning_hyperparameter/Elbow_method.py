# with Elbow method
# output(optimal_k): Optimal number of cluster by data

import numpy as np
from sklearn.cluster import KMeans


def Elbow_Kmeans(data, n_clusters, random_state, n_init):
    kmeans = KMeans(n_clusters, random_state=random_state, n_init=n_init)
    kmeans.fit(data)
    return kmeans


def Elbow_choose_clustering_algorithm(data, clustering_algorithm, n_clusters):
    if clustering_algorithm == 'Kmeans':
        clustering = Elbow_Kmeans(n_clusters, random_state, n_init)
        return clustering
    


def Elbow_method(data, clustering_algorithm, max_clusters):
    wcss = []  # Store WCSS by number of clusters
    
    for k in range(1, max_clusters + 1):
        clustering = Elbow_choose_clustering_algorithm(data, clustering_algorithm, k)
        clustering.fit(data)
        wcss.append(clustering.inertia_)
    
    # Rate of change of slope
    second_diff = np.diff(np.diff(wcss))

    # Choose the point with the largest quadratic difference as the optimal k
    optimal_k = np.argmax(second_diff) + 2  # Index calibration (+2 reason: fewer indexes when using np.diff)

    return optimal_k