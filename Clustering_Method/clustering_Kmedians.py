# Input data is 'X'; Hstack processing on feature_list
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn_extra.cluster import KMedoids
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_Kmedians(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'Kmedians', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    kmedians = KMedoids(n_clusters=n_clusters, random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = kmedians.fit_predict(X)

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG KMedians main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG KMedians main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_kmedians = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_Kmedians(data, X, n_clusters, random_state):
    kmedians = KMedoids(n_clusters=n_clusters, random_state=random_state)   # default; random_state=42 in original, now passed

    cluster_labels = kmedians.fit_predict(X)

    # predict_kmedians = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_kmedians))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels, # Model-generated labels (before CNI)
        'n_clusters' : n_clusters, # Number of clusters requested
        'before_labeling' : kmedians # Model object
    }