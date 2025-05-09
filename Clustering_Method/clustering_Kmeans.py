# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional) overall dictionary
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import KMeans
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Loop_elbow_gs import loop_tuning
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_Kmeans(data, X, max_clusters): # main clustering
    clustering_result_dict = loop_tuning(data, X, 'Kmeans', max_clusters)
    n_clusters = clustering_result_dict['optimul_cluster_n']
    best_parameter_dict = clustering_result_dict['best_parameter_dict']

    kmeans = KMeans(n_clusters=n_clusters, random_state=best_parameter_dict['random_state'], n_init=best_parameter_dict['n_init'])
    clusters = kmeans.fit_predict(X)

    # Debug cluster id
    print(f"\n[DEBUG KMeans main_clustering] Param for CNI 'data' - Shape: {data.shape}")
    print(f"[DEBUG KMeans main_clustering] Param for CNI 'data' - Columns: {list(data.columns)}")
    
    print(f"[DEBUG KMeans main_clustering] Array used for clustering 'X' - Shape: {X.shape}")
    # if not hasattr(X, 'columns'):
    #     print(f"[DEBUG KMeans main_clustering] Array used for clustering 'X' (NumPy array) - First 5 cols of first row: {X[0, :5] if X.shape[0] > 0 and X.shape[1] >= 5 else 'N/A or too small'}")
    
    data['cluster'] = clustering_nomal_identify(data, clusters, n_clusters)

    predict_kmeans = data['cluster']

    return {
        'Cluster_labeling': predict_kmeans,
        'Best_parameter_dict': best_parameter_dict
    }


def pre_clustering_Kmeans(data, X, n_clusters, random_state, n_init):
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)   # default; randomm_state=42, n_init=10

    cluster_labels = kmeans.fit_predict(X)

    predict_kmeans = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_kmeans))  # Counting the number of clusters

    return {
        'Cluster_labeling' : predict_kmeans,
        'n_clusters' : num_clusters,
        'before_labeling' : kmeans
    }