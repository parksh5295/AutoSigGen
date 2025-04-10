# Clustering Methods: Gaussian-means
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from scipy.stats import normaltest
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


class GMeans:
    def __init__(self, max_clusters=10, tol=1e-4, random_state=None):
        self.max_clusters = max_clusters
        self.tol = tol
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        self.labels_ = np.zeros(X.shape[0], dtype=int)  # Initial cluster label
        clusters = [(X, 0)]  # Initial cluster list (data, cluster ID)
        cluster_id = 1  # Cluster ID

        while clusters:
            data, cluster_idx = clusters.pop(0)

            # Skip if cluster is too small or nearly identical
            if len(data) < 8 or np.all(np.std(data, axis=0) < 1e-8):
                self.labels_[self.labels_ == cluster_idx] = cluster_id
                cluster_id += 1
                continue

            # Clustering with K-means (k=2)
            kmeans = KMeans(n_clusters=2, tol=self.tol, random_state=self.random_state)
            kmeans.fit(data)
            new_labels = kmeans.labels_

            # Test if each subcluster follows normality
            for new_cluster_id in range(2):
                sub_data = data[new_labels == new_cluster_id]

                if len(sub_data) < 8:  # Not testing clusters that are too small
                    self.labels_[self.labels_ == cluster_idx] = cluster_id
                    cluster_id += 1
                    continue

                # Instead of: _, p_value = normaltest(sub_data)
                # Use a 1D projection:
                sub_data_1d = sub_data.mean(axis=1)
                # _, p_value = normaltest(sub_data)  # Normality test (calculate p-value)
                # _, p_value = normaltest(sub_data_1d)    # Because normaltest() is sensitive, it's safe to only run it on 1D vectors
                _, p_value = normaltest(sub_data[:, 0])  # Use only the first PCA principal component

                if np.any(p_value < 0.001):  # More granularity when regularity is not followed
                    clusters.append((sub_data, cluster_id))
                else:  # Follow regularity to confirm clusters
                    self.labels_[self.labels_ == cluster_idx] = cluster_id
                    print(f"regul cluster ID: {cluster_id}")
                
                cluster_id += 1

        self.cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0) for i in np.unique(self.labels_)])
        return self

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X):
        return self.fit(X).labels_


def clustering_Gmeans(data, X):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        tune_parameters = Grid_search_all(X, 'Gmeans')

        if not tune_parameters or 'Gmeans' not in tune_parameters or not tune_parameters['Gmeans']['best_params']:
            raise ValueError("Grid search for GMeans failed: No best parameters found.")

        best_params = tune_parameters['Gmeans']['best_params']
        parameter_dict = tune_parameters['Gmeans']['all_params']
        parameter_dict.update(best_params)

        # G-means Clustering (using GaussianMixture)
        gmeans = GMeans(random_state=parameter_dict['random_state'], max_clusters=parameter_dict['max_clusters'], tol=parameter_dict['tol'])
        clusters = gmeans.fit_predict(X)
        n_clusters = len(np.unique(clusters))  # Counting the number of clusters
        data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)
        update_pbar(len(data))

    predict_Gmeans = data['cluster']
    
    return {
        'Cluster_labeling': predict_Gmeans,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_Gmeans(data, X, random_state, max_clusters, tol):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # G-means Clustering (using GaussianMixture)
        gmeans = GMeans(random_state, max_clusters, tol)
        clusters = gmeans.fit_predict(X)
        n_clusters = len(np.unique(clusters))  # Counting the number of clusters
        clustering_data = clustering_nomal_identify(data, clusters, n_clusters)
        update_pbar(len(data))

    predict_Gmeans = clustering_data
    
    return {
        'Cluster_labeling': predict_Gmeans,
        'n_clusters' : n_clusters,
        'before_labeling' : gmeans
    }