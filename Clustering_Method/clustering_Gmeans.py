# Clustering Methods: Gaussian-means
# input 'X' is X_reduced or X rows

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import normaltest
from utils.progressing_bar import progress_bar


class GMeans:
    def __init__(self, max_clusters=10, tol=1e-4, random_state=None):
        self.max_clusters = max_clusters
        self.tol = tol
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0], dtype=int)  # Initial cluster label
        clusters = [(X, 0)]  # Initial cluster list (data, cluster ID)
        cluster_id = 1  # Cluster ID

        while clusters:
            data, cluster_idx = clusters.pop(0)

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

                _, p_value = normaltest(sub_data)  # Normality test (calculate p-value)

                if p_value < 0.05:  # More granularity when regularity is not followed
                    clusters.append((sub_data, cluster_id))
                else:  # Follow regularity to confirm clusters
                    self.labels_[self.labels_ == cluster_idx] = cluster_id
                
                cluster_id += 1

        self.cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0) for i in np.unique(self.labels_)])
        return self

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X):
        return self.fit(X).labels_


def clustering_Gmeans(data, X, random_state):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # G-means Clustering (using GaussianMixture)
        gmeans = GMeans(random_state=random_state)
        cluster_labels = gmeans.fit_predict(X)
        data['cluster'] = cluster_labels
        update_pbar(len(data))

    predict_Gmeans = data['cluster']

    return predict_Gmeans