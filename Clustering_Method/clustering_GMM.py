# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_GMM_normal(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, random_state=parameter_dict['random_state'])
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-normal main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-normal main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_full(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='full', random_state=parameter_dict['random_state'])
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-full main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-full main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
        
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_tied(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='tied', random_state=parameter_dict['random_state'])
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-tied main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-tied main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
        
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_diag(data, X, max_clusters, aligned_original_labels):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm, clusters = fit_gmm_with_retry(X, n_clusters, covariance_type='diag', random_state=parameter_dict['random_state'])

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG GMM-diag main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG GMM-diag main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, n_clusters)

    # predict_GMM = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM(data, X, max_clusters, GMM_type, aligned_original_labels):
    if GMM_type == 'normal':
        predict_GMM_dict = clustering_GMM_normal(data, X, max_clusters, aligned_original_labels)
    elif GMM_type == 'full':
        predict_GMM_dict = clustering_GMM_full(data, X, max_clusters, aligned_original_labels)
    elif GMM_type == 'tied':
        predict_GMM_dict = clustering_GMM_tied(data, X, max_clusters, aligned_original_labels)
    elif GMM_type == 'diag':
        predict_GMM_dict = clustering_GMM_diag(data, X, max_clusters, aligned_original_labels)
    else:
        print("GMM type Error!! -In Clustering")

    predict_GMM = predict_GMM_dict['Cluster_labeling']
    parameter_dict = predict_GMM_dict['Best_parameter_dict']
    
    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


# Precept Function for Clustering Count Tuning Loop

def pre_clustering_GMM_normal(data, X, n_clusters, random_state):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, random_state=random_state) # GMM_normal is usually spherical

    # predict_GMM_normal = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_normal))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : gmm
    }


def pre_clustering_GMM_full(data, X, n_clusters, random_state):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='full', random_state=random_state)

    # predict_GMM_full = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_full))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_tied(data, X, n_clusters, random_state):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='tied', random_state=random_state)

    # predict_GMM_tied = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_tied))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_diag(data, X, n_clusters, random_state):
    gmm, cluster_labels = fit_gmm_with_retry(X, n_clusters, covariance_type='diag', random_state=random_state)

    # predict_GMM_diag = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_GMM_diag))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM(data, X, n_clusters, random_state, GMM_type):
    if GMM_type == 'normal':
        clustering_gmm = pre_clustering_GMM_normal(data, X, n_clusters, random_state)
    elif GMM_type == 'full':
        clustering_gmm = pre_clustering_GMM_full(data, X, n_clusters, random_state)
    elif GMM_type == 'tied':
        clustering_gmm = pre_clustering_GMM_tied(data, X, n_clusters, random_state)
    elif GMM_type == 'diag':
        clustering_gmm = pre_clustering_GMM_diag(data, X, n_clusters, random_state)
    else:
        print("GMM type Error!! -In Clustering")
        return None # Or raise an error
    
    return clustering_gmm


# Functions to automatically update reg_covar to avoid errors
def fit_gmm_with_retry(X, n_components, covariance_type='full', random_state=None, max_reg_covar=100):
    reg_covar = 1e-6
    while reg_covar <= max_reg_covar:
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
                reg_covar=reg_covar
            )
            cluster_labels = gmm.fit_predict(X)
            return gmm, cluster_labels
        except ValueError as e:
            print(f"[Warning] GMM ({covariance_type}) failed with reg_covar={reg_covar:.1e}: {e}")
            reg_covar *= 10

    raise ValueError(f"GMM ({covariance_type}) failed after trying reg_covar up to {max_reg_covar}")
