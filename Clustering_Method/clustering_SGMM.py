# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_SGMM(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'SGMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = sgmm.fit_predict(X)

    # Debug cluster id
    print(f"\n[DEBUG SGMM main_clustering] Param for CNI 'data' - Shape: {data.shape}")
    print(f"[DEBUG SGMM main_clustering] Param for CNI 'data' - Columns: {list(data.columns)}")
    
    print(f"[DEBUG SGMM main_clustering] Array used for clustering 'X' - Shape: {X.shape}")
    # if not hasattr(X, 'columns'):
    #     print(f"[DEBUG SGMM main_clustering] Array used for clustering 'X' (NumPy array) - First 5 cols of first row: {X[0, :5] if X.shape[0] > 0 and X.shape[1] >= 5 else 'N/A or too small'}")
    
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_SGMM = data['cluster']

    return {
        'Cluster_labeling': predict_SGMM,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_SGMM(data, X, n_clusters, random_state):
    sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=random_state)   # default; randomm_state=42

    cluster_labels = sgmm.fit_predict(X)

    predict_SGMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_SGMM,
        'n_clusters' : num_clusters,
        'before_labeling' : sgmm
    }