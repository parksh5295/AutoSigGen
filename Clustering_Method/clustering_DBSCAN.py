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


def clustering_DBSCAN(data, X_reduced_features, original_labels_aligned):
    parameter_dict = {
        'eps': 0.5,
        'count_samples': 5  # Used as min_samples in DBSCAN
    }

    # Hyperparameter Tuning for DBSCAN
    # Grid_search_all takes X, clustering_algorithm, and parameter_dict as arguments.
    # pass a copy to avoid modifying the original parameter_dict
    grid_search_results = Grid_search_all(X_reduced_features, 'DBSCAN', parameter_dict.copy()) 
    
    # grid_search_results is a dictionary with algorithm name as key.
    # Get the best_params from the DBSCAN results.
    dbscan_results = grid_search_results.get('DBSCAN', {})
    best_params = dbscan_results.get('best_params')

    # Update parameter_dict only if best_params is found.
    if best_params is not None:
        parameter_dict.update(best_params)
        print(f"DBSCAN: Updated parameters with grid search results: {best_params}")
    else:
        print("Warning: best_params for DBSCAN not found in grid search results. Using default parameters from initial parameter_dict.")

    # Get eps and min_samples (count_samples) from parameter_dict.
    # GridSearch optimizes for 'min_samples', so we check that key first.
    eps = parameter_dict.get('eps', 0.5) 
    min_samples_from_grid = parameter_dict.get('min_samples') # GridSearch can return 'min_samples'

    if min_samples_from_grid is not None:
        count_samples = min_samples_from_grid
    else:
        count_samples = parameter_dict.get('count_samples', 5) # Initial or updated count_samples
    
    # Perform DBSCAN Clustering
    # clustering_DBSCAN_clustering function takes data, X, eps, count_samples as arguments.
    predict_DBSCAN, num_clusters_actual, dbscan_model = clustering_DBSCAN_clustering(data, X_reduced_features, eps, count_samples)
    
    # Identify Clustering results as normal/abnormal
    # clustering_nomal_identify function takes X, original_labels_aligned, cluster_labels, n_clusters as arguments.
    final_cluster_labels_from_cni = clustering_nomal_identify(X_reduced_features, original_labels_aligned, predict_DBSCAN, num_clusters_actual)
    num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
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