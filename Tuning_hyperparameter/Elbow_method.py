# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Clustering_Method.clustering_kmeans import pre_clustering_kmeans
from Clustering_Method.clustering_kmedians import pre_clustering_kmedians
from Clustering_Method.clustering_GMM import pre_clustering_GMM
from Clustering_Method.clustering_SGMM import pre_clustering_SGMM
from Clustering_Method.clustering_FCM import pre_clustering_FCM
from Clustering_Method.clustering_CK import pre_clustering_CK


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict):   # X: Encoding and embedding, post-PCA, post-delivery
    if clustering_algorithm == 'Kmeans':
        a, b, clustering = pre_clustering_kmeans(data, X, n_clusters, random_state=parameter_dict['random_state'], n_init=parameter_dict['n_init'])

    elif clustering_algorithm == 'Kmedians':
        a, b, clustering = pre_clustering_kmedians(data, X, n_clusters, random_state=parameter_dict['random_state'])

    elif clustering_algorithm == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        a, b, clustering = pre_clustering_GMM(data, X, n_clusters, random_state=parameter_dict['random_state'], GMM_type=GMM_type)

    elif clustering_algorithm == 'SGMM':
        a, b, clustering = pre_clustering_SGMM(data, X, n_clusters, random_state=parameter_dict['random_state'])

    elif clustering_algorithm == 'FCM':
        a, b, clustering = pre_clustering_FCM(data, X, n_clusters)

    elif clustering_algorithm == 'CK':
        a, b, clustering = pre_clustering_CK(data, X, n_clusters)

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict):
    wcss = []  # Store WCSS by number of clusters
    
    for k in range(1, max_clusters + 1):
        clustering = Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, k, parameter_dict)
        clustering.fit(data)
        wcss.append(clustering.inertia_)
    
    # Rate of change of slope
    second_diff = np.diff(np.diff(wcss))

    # Choose the point with the largest quadratic difference as the optimal k
    optimal_k = np.argmax(second_diff) + 2  # Index calibration (+2 reason: fewer indexes when using np.diff)

    return optimal_k    # Appropriate number of clusters