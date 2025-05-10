# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import DBSCAN
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_DBSCAN_clustering(data, X, eps, count_samples):  # Fundamental DBSCAN clustering
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=count_samples) # default; eps=0.5, min_samples=5
    # eps: Radius, min_samples: Minimum number of samples to qualify as a cluster
    clusters = dbscan.fit_predict(X)

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters, dbscan


def clustering_DBSCAN(data, X, aligned_original_labels):
    tune_parameters = Grid_search_all(X, 'DBSCAN')
    best_params = tune_parameters['DBSCAN']['best_params']
    parameter_dict = tune_parameters['DBSCAN']['all_params']
    parameter_dict.update(best_params)

    clusters, num_clusters, dbscan = clustering_DBSCAN_clustering(data, X, eps=parameter_dict['eps'], count_samples=parameter_dict['count_samples'])

    # Debug cluster id (data refers to original data, X is the data used for clustering)
    print(f"\n[DEBUG DBSCAN main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG DBSCAN main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    # The result from CNI is directly used.
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    # predict_DBSCAN = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_DBSCAN(data, X, eps, count_samples):
    # cluster_labels are model-generated labels, num_clusters_actual is the count of unique labels found by DBSCAN
    cluster_labels, num_clusters_actual, dbscan = clustering_DBSCAN_clustering(data, X, eps, count_samples)
    
    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_actual, # Actual n_clusters from DBSCAN
        'before_labeling' : dbscan
    }