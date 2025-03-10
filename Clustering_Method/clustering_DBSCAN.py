# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import DBSCAN
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_DBSCAN_clustering(data, X, eps, count_samples):  # Fundamental DBSCAN clustering
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=count_samples) # default; eps=0.5, min_samples=5
        # eps: Radius, min_samples: Minimum number of samples to qualify as a cluster
        clusters = dbscan.fit_predict(X)
        update_pbar(len(data))
    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters, dbscan


def clustering_DBSCAN(data, X):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        tune_parameters = Grid_search_all(X, 'DBSCAN')
        best_params = tune_parameters['DBSCAN']['best_params']
        parameter_dict = tune_parameters['DBSCAN']['all_params']
        parameter_dict.update(best_params)

        clusters, num_clusters, dbscan = clustering_DBSCAN_clustering(data, X, eps=parameter_dict['eps'], count_samples=parameter_dict['count_samples'])
        data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

        update_pbar(len(data))

    predict_DBSCAN = data['cluster']

    return {
        'Cluster_labeling': predict_DBSCAN,
        'Best_parameter_dict': parameter_dict
    }