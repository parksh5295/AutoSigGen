# A clustering module that uses default values, where each parameter is not optimized
# a control group for this clustering module.
# Output: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}; Name: Clustering

from Clustering_Method.clustering_kmeans import pre_clustering_kmeans
from Clustering_Method.clustering_kmedians import pre_clustering_kmedians
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


def choose_clustering_algorithm_Non_optimization(data, X, clustering_algorithm):
    parameter_dict = {'random_state' : 42, 'n_init' : 30, 'max_clusters' : 1000, 'tol' : 1e-4, 'eps' : 0.5, 'count_samples' : 5,
                        'quantile' : 0.2, 'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2,
                        'max_edge_age' : 50, 'epochs' : 300, 'batch_size' : 256, 'n_neighbors' : 5, 'n_clusters' : 1000
                        }
    
    if clustering_algorithm == 'Kmeans':
        clustering = pre_clustering_kmeans(data, X, parameter_dict['n_clusters'], parameter_dict['random_state'], parameter_dict['n_init'])

    elif clustering_algorithm == 'Kmedians':
        clustering = pre_clustering_kmedians(data, X, parameter_dict['n_clusters'], parameter_dict['random_state'])

    elif clustering_algorithm == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        clustering = pre_clustering_GMM(data, X, parameter_dict['random_state'], parameter_dict['n_clusters'], GMM_type=GMM_type)

    elif clustering_algorithm == 'SGMM':
        clustering = pre_clustering_SGMM(data, X, parameter_dict['random_state'], parameter_dict['n_clusters'])

    elif clustering_algorithm == 'Gmeans':
        clustering = pre_clustering_Gmeans(data, X, parameter_dict['random_state'], parameter_dict['max_clusters'], parameter_dict['tol'])

    elif clustering_algorithm == 'Xmeans':
        clustering = pre_clustering_Xmeans(data, X, parameter_dict['random_state'], parameter_dict['max_clusters'])

    elif clustering_algorithm == 'DBSCAN':
        clustering = pre_clustering_DBSCAN(data, X, parameter_dict['eps'], parameter_dict['count_samples'])

    elif clustering_algorithm == 'MShift':
        clustering = pre_clustering_MShift(data, X, parameter_dict['random_state'], parameter_dict['quantile'], parameter_dict['n_samples'])

    elif clustering_algorithm == 'FCM':
        clustering = pre_clustering_FCM(data, X, parameter_dict['n_clusters'])

    elif clustering_algorithm == 'CK':
        clustering = pre_clustering_CK(data, X, parameter_dict['n_clusters'])

    elif clustering_algorithm == 'NeuralGas':
        clustering = pre_clustering_NeuralGas(data, X, parameter_dict['n_start_nodes'], parameter_dict['max_nodes'], parameter_dict['step'], parameter_dict['max_edge_age'])

    elif clustering_algorithm == 'CANNwKNN':
        clustering = pre_clustering_CANNwKNN(data, X, parameter_dict['epochs'], parameter_dict['batch_size'], parameter_dict['n_neighbors'])

    return clustering