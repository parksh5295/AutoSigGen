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
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        after_elbow = Elbow_method(data, X, 'SGMM', max_clusters)
        n_clusters = after_elbow['optimul_cluster_n']
        parameter_dict = after_elbow['parameter_dict']

        sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=parameter_dict['random_state'])   # default; randomm_state=42
    
        clusters = sgmm.fit_predict(X)
        data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)
        update_pbar(len(data))

    predict_SGMM = data['cluster']

    return {
        'Cluster_labeling': predict_SGMM,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_SGMM(data, X, n_clusters, state):
    sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = sgmm.fit_predict(X)
        update_pbar(len(data))

    predict_SGMM = data['cluster']
    num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return predict_SGMM, num_clusters, sgmm