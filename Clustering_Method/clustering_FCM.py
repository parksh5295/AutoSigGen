# Clustering Methods: Fuzzy C-means
# input 'X' is X_reduced or X rows
# Return: Cluster Information, num_clusters(result), Cluster Information(not fit, optional)

import numpy as np
import skfuzzy as fuzz
from utils.progressing_bar import progress_bar


def clustering_FCM(data, X, n_clusters):
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