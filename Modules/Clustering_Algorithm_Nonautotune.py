# A clustering module that uses default values, where each parameter is not optimized
# a control group for this clustering module.
# Output: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}; Name: Clustering

from Clustering_Method.clustering_Kmeans import pre_clustering_Kmeans
from Clustering_Method.clustering_Kmedians import pre_clustering_Kmedians
from Clustering_Method.clustering_GMM import pre_clustering_GMM
from Clustering_Method.clustering_SGMM import pre_clustering_SGMM
from Clustering_Method.clustering_Gmeans import pre_clustering_Gmeans
from Clustering_Method.clustering_Xmeans import pre_clustering_Xmeans
from Clustering_Method.clustering_DBSCAN import pre_clustering_DBSCAN
from Clustering_Method.clustering_Mshift import pre_clustering_MShift
from Clustering_Method.clustering_FCM import pre_clustering_FCM
from Clustering_Method.clustering_CK import pre_clustering_CK
from Clustering_Method.clustering_NeuralGas import pre_clustering_NeuralGas
from Clustering_Method.clustering_CANNwKNN import pre_clustering_CANNwKNN


def choose_clustering_algorithm_Non_optimization(data, X_reduced_features, original_labels_aligned, clustering_algorithm_choice):
    parameter_dict = {'random_state' : 42, 'n_init' : 30, 'max_clusters' : 1000, 'tol' : 1e-4, 'eps' : 0.5, 'count_samples' : 5,
                        'quantile' : 0.2, 'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2,
                        'max_edge_age' : 50, 'epochs' : 300, 'batch_size' : 256, 'n_neighbors' : 5, 'n_clusters' : 1000
                        }

    GMM_type = None
    clustering = None # Initialize clustering variable

    # Pass original_labels_aligned to each pre_clustering function call.
    # The 'data' and 'X_reduced_features' arguments are passed as required by pre_clustering functions.

    if clustering_algorithm_choice in ['Kmeans', 'kmeans']:
        clustering = pre_clustering_Kmeans(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], parameter_dict['n_init'], original_labels_aligned)

    elif clustering_algorithm_choice in ['Kmedians', 'kmedians']:
        clustering = pre_clustering_Kmedians(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], original_labels_aligned)

    elif clustering_algorithm_choice == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        # Corrected argument order for pre_clustering_GMM
        clustering = pre_clustering_GMM(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], GMM_type, original_labels_aligned)

    elif clustering_algorithm_choice == 'SGMM':
        clustering = pre_clustering_SGMM(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], original_labels_aligned)

    elif clustering_algorithm_choice in ['Gmeans', 'gmeans']:
        clustering = pre_clustering_Gmeans(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['max_clusters'], parameter_dict['tol'], original_labels_aligned)

    elif clustering_algorithm_choice in ['Xmeans', 'xmeans']:
        clustering = pre_clustering_Xmeans(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['max_clusters'], original_labels_aligned)

    elif clustering_algorithm_choice == 'DBSCAN':
        clustering = pre_clustering_DBSCAN(data, X_reduced_features, parameter_dict['eps'], parameter_dict['count_samples'], original_labels_aligned)

    elif clustering_algorithm_choice == 'MShift':
        clustering = pre_clustering_MShift(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['quantile'], parameter_dict['n_samples'], original_labels_aligned)

    elif clustering_algorithm_choice == 'FCM':
        clustering = pre_clustering_FCM(data, X_reduced_features, parameter_dict['n_clusters'], original_labels_aligned)

    elif clustering_algorithm_choice == 'CK':
        clustering = pre_clustering_CK(data, X_reduced_features, parameter_dict['n_clusters'], original_labels_aligned)

    elif clustering_algorithm_choice == 'NeuralGas':
        clustering = pre_clustering_NeuralGas(data, X_reduced_features, parameter_dict['n_start_nodes'], parameter_dict['max_nodes'], parameter_dict['step'], parameter_dict['max_edge_age'], original_labels_aligned)

    elif clustering_algorithm_choice in ['CANNwKNN', 'CANN']:
        print(f"[INFO] CANNwKNN/CANN selected for Non-optimization. Passing original_labels_aligned for consistency.")
        clustering = pre_clustering_CANNwKNN(data, X_reduced_features, parameter_dict['epochs'], parameter_dict['batch_size'], parameter_dict['n_neighbors'], original_labels_aligned) # Added original_labels_aligned

    else:
        print("Unsupported algorithm")
        raise Exception("Unsupported clustering algorithms")
    
    if clustering is None:
        raise Exception(f"Clustering result is None for algorithm: {clustering_algorithm_choice} in Non-optimization mode")

    return clustering, GMM_type