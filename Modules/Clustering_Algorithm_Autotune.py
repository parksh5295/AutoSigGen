# Modules for determining how to cluster
# Output: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}; Name: Clustering

from Clustering_Method.clustering_Kmeans import clustering_Kmeans
from Clustering_Method.clustering_Kmedians import clustering_Kmedians
from Clustering_Method.clustering_GMM import clustering_GMM
from Clustering_Method.clustering_SGMM import clustering_SGMM
from Clustering_Method.clustering_Gmeans import clustering_Gmeans
from Clustering_Method.clustering_Xmeans import clustering_Xmeans
from Clustering_Method.clustering_DBSCAN import clustering_DBSCAN
from Clustering_Method.clustering_Mshift import clustering_MShift
from Clustering_Method.clustering_FCM import clustering_FCM
from Clustering_Method.clustering_CK import clustering_CK
from Clustering_Method.clustering_NeuralGas import clustering_NeuralGas
from Clustering_Method.clustering_CANNwKNN import clustering_CANNwKNN


def choose_clustering_algorithm(data, X_reduced_features, original_labels_aligned, clustering_algorithm_choice, max_clusters=1000):
    '''
    parameter_dict = {'random_state' : random_state, 'n_init' : n_init, 'max_clusters' : max_clusters, 'tol' : tol, 'eps' : eps,
                        'count_samples' : count_samples, 'quantile' : quantile, 'n_samples' : n_samples, 'n_start_nodes' : n_start_nodes,
                        'max_nodes' : max_nodes, 'step' : step, 'max_edge_age' : max_edge_age, 'epochs' : epochs,
                        'batch_size': batch_size, 'n_neighbors' : n_neighbors
    }
    '''
    GMM_type = None
    clustering = None # Initialize clustering variable

    # Pass original_labels_aligned to each clustering function call.
    # The 'data' argument is kept as it might be used by Elbow_method or other hyperparameter tuning logic.
    # X_reduced_features is the actual input for clustering.

    if clustering_algorithm_choice in ['Kmeans', 'kmeans']:
        clustering = clustering_Kmeans(data, X_reduced_features, max_clusters, original_labels_aligned)

    elif clustering_algorithm_choice in ['Kmedians', 'kmedians']:
        clustering = clustering_Kmedians(data, X_reduced_features, max_clusters, original_labels_aligned)

    elif clustering_algorithm_choice == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        clustering = clustering_GMM(data, X_reduced_features, max_clusters, GMM_type, original_labels_aligned)

    elif clustering_algorithm_choice == 'SGMM':
        clustering = clustering_SGMM(data, X_reduced_features, max_clusters, original_labels_aligned)

    elif clustering_algorithm_choice in ['Gmeans', 'gmeans']:
        clustering = clustering_Gmeans(data, X_reduced_features, original_labels_aligned)

    elif clustering_algorithm_choice in ['Xmeans', 'xmeans']:
        clustering = clustering_Xmeans(data, X_reduced_features, original_labels_aligned)

    elif clustering_algorithm_choice == 'DBSCAN':
        clustering = clustering_DBSCAN(data, X_reduced_features, original_labels_aligned)

    elif clustering_algorithm_choice == 'MShift':
        clustering = clustering_MShift(data, X_reduced_features, original_labels_aligned)

    elif clustering_algorithm_choice == 'FCM':
        clustering = clustering_FCM(data, X_reduced_features, max_clusters, original_labels_aligned)

    elif clustering_algorithm_choice == 'CK':
        clustering = clustering_CK(data, X_reduced_features, max_clusters, original_labels_aligned)

    elif clustering_algorithm_choice == 'NeuralGas':
        clustering = clustering_NeuralGas(data, X_reduced_features, original_labels_aligned)

    elif clustering_algorithm_choice in ['CANNwKNN', 'CANN']:
        print(f"[INFO] CANNwKNN/CANN selected. Passing original_labels_aligned for consistency, though it might not be used by its internal logic if CNI is bypassed.")
        clustering = clustering_CANNwKNN(data, X_reduced_features, original_labels_aligned)

    else:
        print("Unsupported algorithm")
        raise Exception("Unsupported clustering algorithms")

    if clustering is None:
        # This check might be problematic if CANNwKNN is expected to return None or a different structure not caught by this.
        # However, CANNwKNN was modified to return a dict similar to others.
        raise Exception(f"Clustering result is None for algorithm: {clustering_algorithm_choice}")

    return clustering, GMM_type