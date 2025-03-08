# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# Return: Cluster Information, num_clusters(result), Cluster Information(not fit, optional)

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_SGMM(data, X, n_clusters, state):
    sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = sgmm.fit_predict(X)
        update_pbar(len(data))

    predict_SGMM = data['cluster']
    num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return predict_SGMM, num_clusters, sgmm


def pre_clustering_SGMM(data, X, n_clusters, state):
    sgmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = sgmm.fit_predict(X)
        update_pbar(len(data))

    predict_SGMM = data['cluster']
    num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return predict_SGMM, num_clusters, sgmm