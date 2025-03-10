# Clustering Methods: Fuzzy C-means
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
import skfuzzy as fuzz
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_FCM(data, X, max_clusters):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        after_elbow = Elbow_method(data, X, 'FCM', max_clusters)
        n_clusters = after_elbow['optimul_cluster_n']
        parameter_dict = after_elbow['parameter_dict']

        # Fuzzy C-Means Clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
        )

        # Assign clusters based on maximum membership
        cluster_labels = np.argmax(u, axis=0)
        data['cluster'] = clustering_nomal_identify(data, cluster_labels, n_clusters)
    update_pbar(len(data))

    predict_FCM = data['cluster']

    return {
        'Cluster_labeling': predict_FCM,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_FCM(data, X, n_clusters):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Fuzzy C-Means Clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
        )

        # Assign clusters based on maximum membership
        cluster_labels = np.argmax(u, axis=0)
        data['cluster'] = cluster_labels
    update_pbar(len(data))

    predict_FCM = data['cluster']
    num_clusters = len(np.unique(predict_FCM))  # Counting the number of clusters

    return predict_FCM, num_clusters, cluster_labels