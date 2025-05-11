# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_SGMM(data, X, max_clusters, original_labels_aligned):
    # Define an initial parameter_dict that includes reg_covar
    # This will be passed to Elbow_method, which might use its own default if this is None,
    # but it's good practice to define it here for clarity and control.
    initial_parameter_dict = {
        'random_state': 42,
        'reg_covar': 1e-6  # Default regularization
        # Add other parameters expected by Elbow_method or pre_clustering_SGMM if necessary
    }

    # Elbow_method now expects a parameter_dict and will use reg_covar from it for SGMM
    after_elbow = Elbow_method(data, X, 'SGMM', max_clusters, parameter_dict=initial_parameter_dict.copy())
    n_clusters = after_elbow['optimul_cluster_n']
    
    # The parameter_dict returned by Elbow_method should contain the used random_state and reg_covar
    parameter_dict_from_elbow = after_elbow['best_parameter_dict']

    # Apply Spherical GMM (SGMM) Clustering with tuned n_clusters and reg_covar
    sgmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=parameter_dict_from_elbow.get('random_state', 42), # Ensure random_state is used
        reg_covar=parameter_dict_from_elbow.get('reg_covar', 1e-6)    # Ensure reg_covar is used
    )
    
    cluster_labels = sgmm.fit_predict(X)
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG SGMM main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG SGMM main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, original_labels_aligned, cluster_labels, n_clusters)
    num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict_from_elbow
    }


def pre_clustering_SGMM(data, X, n_clusters, random_state, reg_covar=1e-6): # Added reg_covar
    # Default reg_covar is 1e-6, similar to sklearn's default for GaussianMixture
    sgmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=random_state,
        reg_covar=reg_covar  # Pass reg_covar to the model
    )

    cluster_labels = sgmm.fit_predict(X)

    # predict_SGMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : sgmm
    }