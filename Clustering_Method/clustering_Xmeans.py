# input 'X' is X_reduced or X rows
# Clustering Algorithm: X-means; Autonomously tuning n_clusters in k-means

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.progressing_bar import progress_bar
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify

# Y-Means Clustering Function
def x_means_clustering(X, max_clusters):
    best_score = -1
    best_model = None
    best_k = 2
    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_model = model
            best_k = k
    return best_model, best_k

def clustering_Xmeans_clustering(data, max_clusters, X):  # Fundamental Xmeans clustering
    # default; max=clusters=10, 
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Perform Y-Means Clustering
        model, optimal_k = x_means_clustering(X, max_clusters)
        clusters = model.labels_

        update_pbar(len(data))

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters


def clustering_Xmeans(data, max_clusters, X):
    clusters, num_clusters = clustering_Xmeans_clustering(data, max_clusters, X)
    data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

    return