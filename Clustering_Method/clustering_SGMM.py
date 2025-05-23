# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_GMM import fit_gmm_with_retry # Import the retry mechanism from GMM


# fit_sgmm_with_retry function is now removed, using fit_gmm_with_retry from clustering_GMM.py

def clustering_SGMM(data, X, max_clusters, original_labels_aligned):
    initial_parameter_dict = {
        'random_state': 42,
        'reg_covar': 1e-6  # Base reg_covar for Elbow_method and initial suggestion for final fit
    }

    # Elbow_method now expects a parameter_dict and will use reg_covar from it for SGMM
    after_elbow = Elbow_method(data, X, 'SGMM', max_clusters, parameter_dict=initial_parameter_dict.copy())
    n_clusters = after_elbow['optimul_cluster_n']
    
    # The parameter_dict returned by Elbow_method should contain the used random_state and reg_covar
    parameter_dict_from_elbow = after_elbow['best_parameter_dict']

    # Use fit_gmm_with_retry for the final model fitting, specifying spherical covariance
    sgmm_model, cluster_labels = fit_gmm_with_retry(
        X, 
        n_components=n_clusters, 
        covariance_type='spherical', # Specify spherical for SGMM
        random_state=parameter_dict_from_elbow.get('random_state', 42)
        # initial_reg_covar is handled internally by fit_gmm_with_retry (starts at 1e-6)
        # max_reg_covar and retry_multiplier are also internal to fit_gmm_with_retry
    )
    
    print(f"\n[DEBUG SGMM main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    final_cluster_labels_from_cni = clustering_nomal_identify(X, original_labels_aligned, cluster_labels, n_clusters)

    # Update parameter_dict_from_elbow with the actually used reg_covar
    parameter_dict_from_elbow['reg_covar'] = sgmm_model.reg_covar_

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict_from_elbow
    }


def pre_clustering_SGMM(data, X, n_clusters, random_state, reg_covar=1e-6):
    # Use fit_gmm_with_retry here as well, specifying spherical covariance
    # The input reg_covar to this function is not directly used as initial_reg_covar by fit_gmm_with_retry,
    # as fit_gmm_with_retry starts its own loop at 1e-6. This is consistent with GMM's usage.
    try:
        sgmm_model, cluster_labels = fit_gmm_with_retry(
            X,
            n_components=n_clusters,
            covariance_type='spherical', # Specify spherical for SGMM
            random_state=random_state
        )
    except ValueError as e:
        print(f"[Error] pre_clustering_SGMM: fit_gmm_with_retry (spherical) failed for k={n_clusters}. Error: {e}")
        return {
            'model_labels': None, 
            'n_clusters': n_clusters,
            'before_labeling': None
        }

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, 
        'before_labeling' : sgmm_model
    }