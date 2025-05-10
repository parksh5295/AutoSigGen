# input 'X' is X_reduced or X rows
# Clustering Method: MeanShift
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MeanShift, estimate_bandwidth
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_MShift_clustering(data, X, state, quantile, n_samples):  # Fundamental MeanShift clustering
    # Estimate bandwidth based on the data
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, random_state=state) # default; randomm_state=42, n_samples=500, quantile=0.2
    if bandwidth <= 0:
        bandwidth = 0.1  # Minimum safe value
    
    # Apply MeanShift with the estimated bandwidth
    MShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clusters = MShift.fit_predict(X)

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters
    
    return clusters, num_clusters, MShift


def clustering_MShift(data, X, aligned_original_labels):
    tune_parameters = Grid_search_all(X, 'MShift')
    best_params = tune_parameters['MShift']['best_params']
    parameter_dict = tune_parameters['MShift']['all_params']
    parameter_dict.update(best_params)

    clusters, num_clusters, MShift = clustering_MShift_clustering(data, X, state=parameter_dict['random_state'], quantile=parameter_dict['quantile'], n_samples=parameter_dict['n_samples'])

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG MeanShift main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG MeanShift main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    # predict_MShift = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


# Additional classes for Grid Search
class MeanShiftWithDynamicBandwidth(BaseEstimator, ClusterMixin):
    def __init__(self, quantile=0.3, n_samples=500, bin_seeding=True):
        self.quantile = quantile
        self.n_samples = n_samples
        self.bin_seeding = bin_seeding
        self.bandwidth = None
        self.model = None

    def fit(self, X, y=None):
        # Dynamically set bandwidth based on data
        self.bandwidth = estimate_bandwidth(X, quantile=self.quantile, n_samples=self.n_samples)

        # Set a stable minimum
        if self.bandwidth < 1e-3:
            print(f"Estimated bandwidth too small ({self.bandwidth:.5f}) → Adjusted to 0.001")
            self.bandwidth = 1e-3

        self.model = MeanShift(bandwidth=self.bandwidth, bin_seeding=self.bin_seeding)
        self.model.fit(X)

        self.labels_ = self.model.labels_
        
        return self

    def predict(self, X):
        return self.model.predict(X)
    

def pre_clustering_MShift(data, X, random_state, quantile, n_samples):
    # cluster_labels are model-generated labels, num_clusters_actual is the count of unique labels found by MeanShift
    cluster_labels, num_clusters_actual, MShift = clustering_MShift_clustering(data, X, random_state, quantile, n_samples)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_actual, # Actual n_clusters from MeanShift
        'before_labeling' : MShift
    }