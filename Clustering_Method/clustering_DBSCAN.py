# input 'X' is X_reduced or X rows
# Return: Cluster Information, num_clusters(result), Cluster Information(not fit, optional)

import numpy as np
from sklearn.cluster import DBSCAN
from utils.progressing_bar import progress_bar
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


def clustering_DBSCAN(data, eps, count_samples, X):
    clusters, num_clusters, dbscan = clustering_DBSCAN_clustering(data, X, eps, count_samples)
    data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)
    predict_DBSCAN = data['cluster']

    return predict_DBSCAN, num_clusters, dbscan