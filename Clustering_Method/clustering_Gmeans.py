# Clustering Methods: Gaussian-means
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from scipy.stats import normaltest
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


class GMeans:
    def __init__(self, max_clusters=10, tol=1e-4, random_state=None):
        self.max_clusters = max_clusters
        self.tol = tol
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.labels_ = -np.ones(X.shape[0], dtype=int)  # Initial cluster label
        clusters = [(np.arange(X.shape[0]), X, 0)]  # Initial cluster list (index_in_X, data, cluster_idx)
        cluster_id = 0  # Cluster ID

        while clusters:
            indices, data, cluster_idx = clusters.pop(0)

            # Skip if cluster is too small or nearly identical
            if len(data) < 8 or np.all(np.std(data, axis=0) < 1e-8):
                self.labels_[indices] = cluster_id
                cluster_id += 1
                continue

            # Clustering with K-means (k=2)
            kmeans = KMeans(n_clusters=2, tol=self.tol, random_state=self.random_state)
            kmeans.fit(data)
            new_labels = kmeans.labels_

            # Test if each subcluster follows normality
            for new_cluster_id in range(2):
                sub_data = data[new_labels == new_cluster_id]

                if len(sub_data) < 8:
                    self.labels_[indices[new_labels == new_cluster_id]] = cluster_id
                    cluster_id += 1
                    continue

                # Instead of: _, p_value = normaltest(sub_data)
                # Use a 1D projection:
                # sub_data_1d = sub_data.mean(axis=1)
                # _, p_value = normaltest(sub_data)  # Normality test (calculate p-value)
                # _, p_value = normaltest(sub_data_1d)    # Because normaltest() is sensitive, it's safe to only run it on 1D vectors
                _, p_value = normaltest(sub_data[:, 0])  # Use only the first PCA principal component

                if np.any(p_value < 0.01):  # More granularity when regularity is not followed
                    new_indices = indices[new_labels == new_cluster_id]
                    clusters.append((new_indices, sub_data, cluster_id))
                else:
                    self.labels_[indices[new_labels == new_cluster_id]] = cluster_id
                
                cluster_id += 1

        self.cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0) for i in np.unique(self.labels_)])
        return self

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X):
        return self.fit(X).labels_


def clustering_Gmeans(data, X, aligned_original_labels):
    tune_parameters = Grid_search_all(X, 'Gmeans')

    if not tune_parameters or 'Gmeans' not in tune_parameters or not tune_parameters['Gmeans']['best_params']:
        raise ValueError("Grid search for GMeans failed: No best parameters found.")

    best_params = tune_parameters['Gmeans']['best_params']
    parameter_dict = tune_parameters['Gmeans']['all_params']
    parameter_dict.update(best_params)

    # G-means Clustering (using GaussianMixture)
    gmeans = GMeans(random_state=parameter_dict['random_state'], max_clusters=parameter_dict['max_clusters'], tol=parameter_dict['tol'])
    clusters = gmeans.fit_predict(X)
    n_clusters = len(np.unique(clusters))  # Counting the number of clusters

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_Gmeans = data['cluster'] # Old way
    
    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_Gmeans(data, X, random_state, max_clusters, tol):
    # G-means Clustering
    gmeans = GMeans(random_state=random_state, max_clusters=max_clusters, tol=tol) # Pass params to GMeans class constructor
    cluster_labels = gmeans.fit_predict(X)
    n_clusters_actual = len(np.unique(cluster_labels))  # Actual number of clusters found by GMeans

    # REMOVED CNI call
    # final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters_actual)
    # num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))
    
    return {
        # 'Cluster_labeling': final_cluster_labels_from_cni,
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters_actual, # Actual n_clusters from GMeans
        'before_labeling' : gmeans
    }