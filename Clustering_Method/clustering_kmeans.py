# input 'X' is X_reduced or X rows

import numpy as np
from sklearn.cluster import KMeans
from utils.progressing_bar import progress_bar


def clustering_kmeans(data, X, n_clusters, random_state, n_init):
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)   # default; randomm_state=42, n_init=10

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = kmeans.fit_predict(X)
        update_pbar(len(data))

    predict_kmeans = data['cluster']
    num_clusters = len(np.unique(predict_kmeans))  # Counting the number of clusters

    
    return predict_kmeans, num_clusters