# Input data is 'X'; Hstack processing on feature_list
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn_extra.cluster import KMedoids
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_Kmedians(data, X, max_clusters):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        after_elbow = Elbow_method(data, X, 'Kmedians', max_clusters)
        n_clusters = after_elbow['optimul_cluster_n']
        parameter_dict = after_elbow['parameter_dict']

        kmedians = KMedoids(n_clusters=n_clusters, random_state=parameter_dict['random_state'])   # default; randomm_state=42

        clusters = kmedians.fit_predict(X)
        data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)
        update_pbar(len(data))

    predict_kmedians = data['cluster']

    return {
        'Cluster_labeling': predict_kmedians,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_Kmedians(data, X, n_clusters, random_state=42):
    kmedians = KMedoids(n_clusters=n_clusters, random_state=random_state)   # default; randomm_state=42

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        cluster_labels = kmedians.fit_predict(X)
        update_pbar(len(data))

    predict_kmedians = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_kmedians))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_kmedians,
        'n_clusters' : num_clusters,
        'before_labeling' : kmedians
    }