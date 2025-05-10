# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional) overall dictionary
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import KMeans
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Loop_elbow_gs import loop_tuning
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_Kmeans(data, X, max_clusters, aligned_original_labels): # main clustering
    clustering_result_dict = loop_tuning(data, X, 'Kmeans', max_clusters)
    n_clusters = clustering_result_dict['optimul_cluster_n']
    best_parameter_dict = clustering_result_dict['best_parameter_dict']

    kmeans = KMeans(n_clusters=n_clusters, random_state=best_parameter_dict['random_state'], n_init=best_parameter_dict['n_init'])
    clusters = kmeans.fit_predict(X)

    # Debug cluster id (data refers to original data, X is the data used for clustering)
    print(f"\n[DEBUG KMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG KMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    # The result from CNI is directly used, not assigned to data['cluster'] here.
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_kmeans = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': best_parameter_dict
    }


def pre_clustering_Kmeans(data, X, n_clusters, random_state, n_init):
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(X)

    # REMOVED call to clustering_nomal_identify
    # final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters)
    # num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        # 'Cluster_labeling' : final_cluster_labels_from_cni, # REMOVED CNI result
        'model_labels' : cluster_labels, # Model-generated labels (before CNI)
        'n_clusters' : n_clusters, # Number of clusters requested (can be different from unique labels in final_cluster_labels_from_cni)
        'before_labeling' : kmeans # Model object
    }