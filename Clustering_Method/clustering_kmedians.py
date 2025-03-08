# Input data is 'X'; Hstack processing on feature_list
# input 'X' is X_reduced or X rows
# Return: Cluster Information, num_clusters(result), Cluster Information(not fit, optional)

import numpy as np
from sklearn_extra.cluster import KMedoids
from utils.progressing_bar import progress_bar


def clustering_kmedians(data, X, n_clusters, state):
    kmedians = KMedoids(n_clusters=n_clusters, random_state=state)   # default; randomm_state=42

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = kmedians.fit_predict(X)
        update_pbar(len(data))

    predict_kmedians = data['cluster']
    num_clusters = len(np.unique(predict_kmedians))  # Counting the number of clusters

    return predict_kmedians, num_clusters, kmedians


def pre_clustering_kmedians(data, X, n_clusters, state):
    kmedians = KMedoids(n_clusters=n_clusters, random_state=state)   # default; randomm_state=42

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = kmedians.fit_predict(X)
        update_pbar(len(data))

    predict_kmedians = data['cluster']
    num_clusters = len(np.unique(predict_kmedians))  # Counting the number of clusters

    return predict_kmedians, num_clusters, kmedians