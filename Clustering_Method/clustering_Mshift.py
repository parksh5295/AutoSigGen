# input 'X' is X_reduced or X rows
# Clustering Method: MeanShift

import numpy as np
from sklearn.cluster import MeanShift,  estimate_bandwidth
from utils.progressing_bar import progress_bar
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_MShift_clustering(data, state, quantile, n_samples, X):  # Fundamental MeanShift clustering
    # Estimate bandwidth based on the data
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, random_state=state) # default; randomm_state=42, n_samples=500, quantile=0.2
    
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Apply MeanShift with the estimated bandwidth
        MShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clusters = MShift.fit_predict(X)
        update_pbar(len(data))
    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters


def clustering_MShift(data, X, state, quantile, n_samples):
    clusters, num_clusters = clustering_MShift_clustering(data, state, quantile, n_samples, X)
    data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

    predict_MShift = data['cluster']

    return predict_MShift