# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Clustering_Method.common_clustering import get_clustering_function


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict):   # X: Encoding and embedding, post-PCA, post-delivery
    pre_clustering_func = get_clustering_function(clustering_algorithm)

    if clustering_algorithm == 'Kmeans':
        clustering = pre_clustering_func(data, X, n_clusters, random_state=parameter_dict['random_state'], n_init=parameter_dict['n_init'])
    elif clustering_algorithm == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        clustering = pre_clustering_func(data, X, n_clusters, random_state=parameter_dict['random_state'], GMM_type=GMM_type)
    else:
        clustering = pre_clustering_func(data, X, n_clusters, random_state=parameter_dict['random_state'])

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict=None):
    if parameter_dict is None:
        parameter_dict = {'random_state' : 42, 'n_init' : 30, 'max_clusters' : 1000, 'tol' : 1e-4, 'eps' : 0.5, 'count_samples' : 5,
                            'quantile' : 0.2, 'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2,
                            'max_edge_age' : 50, 'epochs' : 300, 'batch_size' : 256, 'n_neighbors' : 5
                            }

    wcss = []  # Store WCSS by number of clusters
    
    for k in range(1, max_clusters + 1):
        clustering = Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, k, parameter_dict)
        clustering_before_label = clustering['before_labeling']
        clustering_before_label.fit(data)
        wcss.append(clustering_before_label.inertia_)
    
    # Rate of change of slope
    second_diff = np.diff(np.diff(wcss))

    # Choose the point with the largest quadratic difference as the optimal k
    optimal_k = np.argmax(second_diff) + 2  # Index calibration (+2 reason: fewer indexes when using np.diff)

    return {
        'optimul_cluster_n': optimal_k,    # Appropriate number of clusters
        'parameter_dict': parameter_dict
    }