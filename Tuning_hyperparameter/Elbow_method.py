# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Modules.Clustering_Algorithm import choose_clustering_algorithm


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters):   # X: Encoding and embedding, post-PCA, post-delivery
    parameter_dict = {'random_state' : 42, 'n_init' : 10, 'max_clusters' : 50, 'eps' : 0.5, 'count_samples' : 5, 'quantile' : 0.2,
                      'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2, 'max_edge_age' : 50
                      }

    if clustering_algorithm == 'Kmeans':
        clustering = choose_clustering_algorithm(data, X, 'Kmeans', n_clusters, parameter_dict)

    elif clustering_algorithm == 'Kmedians':
        clustering = choose_clustering_algorithm(data, X, 'Kmedians', n_clusters, parameter_dict)

    elif clustering_algorithm == 'GMM':
        clustering = choose_clustering_algorithm(data, X, 'GMM', n_clusters, parameter_dict)

    elif clustering_algorithm == 'SGMM':
        clustering = choose_clustering_algorithm(data, X, 'SGMM', n_clusters, parameter_dict)

    elif clustering_algorithm == 'FCM':
        clustering = choose_clustering_algorithm(data, X, 'FCM', n_clusters, parameter_dict)

    elif clustering_algorithm == 'CK':
        clustering = choose_clustering_algorithm(data, X, 'CK', n_clusters, parameter_dict)

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters):
    wcss = []  # Store WCSS by number of clusters
    
    for k in range(1, max_clusters + 1):
        clustering = Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, k)
        clustering.fit(data)
        wcss.append(clustering.inertia_)
    
    # Rate of change of slope
    second_diff = np.diff(np.diff(wcss))

    # Choose the point with the largest quadratic difference as the optimal k
    optimal_k = np.argmax(second_diff) + 2  # Index calibration (+2 reason: fewer indexes when using np.diff)

    return optimal_k    # Appropriate number of clusters