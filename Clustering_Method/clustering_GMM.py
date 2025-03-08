# input 'X' is X_reduced or X rows
# Return: Cluster Information, num_clusters(result), Cluster Information(not fit, optional)

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_GMM_normal(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def clustering_GMM_full(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def clustering_GMM_tied(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def clustering_GMM_diag(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def clustering_GMM(data, X, n_clusters, state, GMM_type):
    if GMM_type == 'normal':
        predict_GMM, num_clusters, gmm = clustering_GMM_normal(data, state, X, n_clusters)
    elif GMM_type == 'full':
        predict_GMM, num_clusters, gmm = clustering_GMM_full(data, state, X, n_clusters)
    elif GMM_type == 'tied':
        predict_GMM, num_clusters, gmm = clustering_GMM_tied(data, state, X, n_clusters)
    elif GMM_type == 'diag':
        predict_GMM, num_clusters, gmm = clustering_GMM_diag(data, state, X, n_clusters)
    else:
        print("GMM type Error!! -In Clustering")
    
    return predict_GMM, num_clusters, gmm


# Precept Function for Clustering Count Tuning Loop

def pre_clustering_GMM_normal(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def pre_clustering_GMM_full(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def pre_clustering_GMM_tied(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def pre_clustering_GMM_diag(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return predict_GMM, num_clusters, gmm


def pre_clustering_GMM(data, X, n_clusters, state, GMM_type):
    if GMM_type == 'normal':
        predict_GMM, num_clusters, gmm = pre_clustering_GMM_normal(data, state, X, n_clusters)
    elif GMM_type == 'full':
        predict_GMM, num_clusters, gmm = pre_clustering_GMM_full(data, state, X, n_clusters)
    elif GMM_type == 'tied':
        predict_GMM, num_clusters, gmm = pre_clustering_GMM_tied(data, state, X, n_clusters)
    elif GMM_type == 'diag':
        predict_GMM, num_clusters, gmm = pre_clustering_GMM_diag(data, state, X, n_clusters)
    else:
        print("GMM type Error!! -In Clustering")
    
    return predict_GMM, num_clusters, gmm