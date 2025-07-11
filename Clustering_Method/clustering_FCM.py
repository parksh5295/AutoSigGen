# Clustering Methods: Fuzzy C-means
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
import skfuzzy as fuzz
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_FCM(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'FCM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    # Fuzzy C-Means Clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # Debug cluster id (X is the data used for clustering, X.T is passed to cmeans)
    print(f"\n[DEBUG FCM main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG FCM main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters)

    # predict_FCM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_FCM(data, X, n_clusters):
    # Fuzzy C-Means Clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
    )

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # predict_FCM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_FCM))  # Counting the number of clusters

    # Wrapped FCM Model Classes
    fuzzy_model = FCMFakeModel(cntr, u, fpc)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested (can be different from unique labels in final_cluster_labels_from_cni)
        'before_labeling' : fuzzy_model
    }

class FCMFakeModel:
    def __init__(self, cntr, u, fpc):
        self.cntr = cntr
        self.u = u
        self.fpc = fpc

    def fit(self, X):
        # Do nothing because Fit is already over
        return self

    @property
    def inertia_(self):
        # The higher the FPC, the better, so invert the sign to match the inertia logic
        return -self.fpc

    def predict(self, X_new):
        # argmax after soft prediction with cmeans_predict
        return np.argmax(
            fuzz.cluster.cmeans_predict(X_new.T, self.cntr, m=2, error=0.005, maxiter=1000)[0],
            axis=0
        )
