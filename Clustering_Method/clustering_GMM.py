# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_GMM_normal(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm = GaussianMixture(n_components=n_clusters, random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = gmm.fit_predict(X)
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_GMM = data['cluster']

    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_full(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = gmm.fit_predict(X)
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_GMM = data['cluster']

    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_tied(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = gmm.fit_predict(X)
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_GMM = data['cluster']

    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM_diag(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'GMM', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=parameter_dict['random_state'])   # default; randomm_state=42

    clusters = gmm.fit_predict(X)
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_GMM = data['cluster']

    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


def clustering_GMM(data, X, max_clusters, GMM_type):
    if GMM_type == 'normal':
        predict_GMM_dict = clustering_GMM_normal(data, X, max_clusters)
    elif GMM_type == 'full':
        predict_GMM_dict = clustering_GMM_full(data, X, max_clusters)
    elif GMM_type == 'tied':
        predict_GMM_dict = clustering_GMM_tied(data, X, max_clusters)
    elif GMM_type == 'diag':
        predict_GMM_dict = clustering_GMM_diag(data, X, max_clusters)
    else:
        print("GMM type Error!! -In Clustering")

    predict_GMM = predict_GMM_dict['Cluster_labeling']
    parameter_dict = predict_GMM_dict['Best_parameter_dict']
    
    return {
        'Cluster_labeling': predict_GMM,
        'Best_parameter_dict': parameter_dict
    }


# Precept Function for Clustering Count Tuning Loop

def pre_clustering_GMM_normal(data, X, random_state, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)   # default; randomm_state=42

    cluster_labels = gmm.fit_predict(X)

    predict_GMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_GMM,
        'n_clusters' : num_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_full(data, X, random_state, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=random_state)   # default; randomm_state=42

    cluster_labels = gmm.fit_predict(X)

    predict_GMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_GMM,
        'n_clusters' : num_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_tied(data, X, random_state, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=random_state)   # default; randomm_state=42

    cluster_labels = gmm.fit_predict(X)

    predict_GMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_GMM,
        'n_clusters' : num_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM_diag(data, X, random_state, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=random_state)   # default; randomm_state=42

    cluster_labels = gmm.fit_predict(X)

    predict_GMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_GMM))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_GMM,
        'n_clusters' : num_clusters,
        'before_labeling' : gmm
    }


def pre_clustering_GMM(data, X, n_clusters, random_state, GMM_type):
    if GMM_type == 'normal':
        clustering_gmm = pre_clustering_GMM_normal(data, X, random_state, n_clusters)
    elif GMM_type == 'full':
        clustering_gmm = pre_clustering_GMM_full(data, X, random_state, n_clusters)
    elif GMM_type == 'tied':
        clustering_gmm = pre_clustering_GMM_tied(data, X, random_state, n_clusters)
    elif GMM_type == 'diag':
        clustering_gmm = pre_clustering_GMM_diag(data, X, random_state, n_clusters)
    else:
        print("GMM type Error!! -In Clustering")
    
    return clustering_gmm