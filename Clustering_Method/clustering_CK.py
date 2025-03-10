# Clustering Algorithm: Custafson-Kessel (Similarly to Fuzzy Algorithm)
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


# Gustafson-Kessel Clustering Implementation
def ck_cluster(X, c, m=2, error=0.005, maxiter=1000):
    """
    Gustafson-Kessel Clustering Algorithm.
    
    Parameters:
        X: ndarray
            Input data of shape (n_samples, n_features).
        c: int
            Number of clusters.
        m: float
            Fuzziness coefficient (default=2).
        error: float
            Stopping criterion threshold (default=0.005).
        maxiter: int
            Maximum number of iterations (default=1000).
            
    Returns:
        cntr: ndarray
            Cluster centers of shape (c, n_features).
        u: ndarray
            Final membership matrix of shape (c, n_samples).
        d: ndarray
            Distance matrix of shape (c, n_samples).
        fpc: float
            Final fuzzy partition coefficient.
    """
    n_samples, n_features = X.shape
    u = np.random.dirichlet(np.ones(c), size=n_samples).T  # Random initialization of membership matrix
    cntr = np.zeros((c, n_features))
    cov_matrices = np.array([np.eye(n_features) for _ in range(c)])  # Initial covariance matrices
    d = np.zeros((c, n_samples))

    for iteration in range(maxiter):
        # Calculate cluster centers
        um = u ** m
        cntr = np.dot(um, X) / um.sum(axis=1, keepdims=True)

        # Update covariance matrices
        for i in range(c):
            diff = X - cntr[i]  # diff의 shape: (n_samples, n_features)
            cov_matrices[i] = np.dot((um[i][:, np.newaxis] * diff).T, diff) / um[i].sum()
            cov_matrices[i] /= np.linalg.det(cov_matrices[i]) ** (1 / n_features)
        '''
        # Checking matrix dimensions
        print("um[i].shape:", um[i].shape)
        print("diff.shape:", diff.shape)
        print("cov_matrices[i].shape:", cov_matrices[i].shape)
        '''
        
        # Calculate distances and update membership
        for i in range(c):
            diff = X - cntr[i]
            d[i] = np.sqrt(np.sum(np.dot(diff, np.linalg.inv(cov_matrices[i])) * diff, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid division by zero
        u_new = 1.0 / np.sum((d / d[:, np.newaxis]) ** (2 / (m - 1)), axis=0)

        # Check for convergence
        if np.linalg.norm(u_new - u) < error:
            break
        u = u_new

    fpc = np.sum(u ** m) / n_samples
    return cntr, u, d, fpc


def clustering_CK(data, X, max_clusters):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        after_elbow = Elbow_method(data, X, 'CK', max_clusters)
        n_clusters = after_elbow['optimul_cluster_n']
        parameter_dict = after_elbow['parameter_dict']

        # Perform Gustafson-Kessel Clustering
        cntr, u, d, fpc = ck_cluster(X, c=n_clusters, m=2)

        # Assign clusters based on maximum membership
        cluster_labels = np.argmax(u, axis=0)
        data['cluster'] = clustering_nomal_identify(data, cluster_labels, n_clusters)
    update_pbar(len(data))

    predict_CK = data['cluster']

    return {
        'Cluster_labeling': predict_CK,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_CK(data, X, n_clusters):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Perform Gustafson-Kessel Clustering
        cntr, u, d, fpc = ck_cluster(X, c=n_clusters, m=2)

        # Assign clusters based on maximum membership
        cluster_labels = np.argmax(u, axis=0)
        data['cluster'] = cluster_labels
    update_pbar(len(data))

    predict_CK = data['cluster']
    num_clusters = len(np.unique(predict_CK))  # Counting the number of clusters

    return predict_CK, num_clusters, cluster_labels