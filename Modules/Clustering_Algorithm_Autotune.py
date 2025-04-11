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


def choose_clustering_algorithm(data, X, clustering_algorithm, max_clusters=1000):
    '''
    parameter_dict = {'random_state' : random_state, 'n_init' : n_init, 'max_clusters' : max_clusters, 'tol' : tol, 'eps' : eps,
                        'count_samples' : count_samples, 'quantile' : quantile, 'n_samples' : n_samples, 'n_start_nodes' : n_start_nodes,
                        'max_nodes' : max_nodes, 'step' : step, 'max_edge_age' : max_edge_age, 'epochs' : epochs,
                        'batch_size': batch_size, 'n_neighbors' : n_neighbors
    }
    '''
    GMM_type = None

    if clustering_algorithm in ['Kmeans', 'kmeans']:
        clustering = clustering_Kmeans(data, X, max_clusters)

    elif clustering_algorithm in ['Kmedians', 'kmedians']:
        clustering = clustering_Kmedians(data, X, max_clusters)

    elif clustering_algorithm == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        clustering = clustering_GMM(data, X, max_clusters, GMM_type=GMM_type)

    elif clustering_algorithm == 'SGMM':
        clustering = clustering_SGMM(data, X, max_clusters)

    elif clustering_algorithm in ['Gmeans', 'gmeans']:
        clustering = clustering_Gmeans(data, X)

    elif clustering_algorithm in ['Xmeans', 'xmeans']:
        clustering = clustering_Xmeans(data, X)

    elif clustering_algorithm == 'DBSCAN':
        clustering = clustering_DBSCAN(data, X)

    elif clustering_algorithm == 'MShift':
        clustering = clustering_MShift(data, X)

    elif clustering_algorithm == 'FCM':
        clustering = clustering_FCM(data, X, max_clusters)

    elif clustering_algorithm == 'CK':
        clustering = clustering_CK(data, X, max_clusters)

    elif clustering_algorithm == 'NeuralGas':
        clustering = clustering_NeuralGas(data, X)

    elif clustering_algorithm in ['CANNwKNN', 'CANN']:
        clustering = clustering_CANNwKNN(data, X)

    else:
        print("Unsupported algorithm")
        raise Exception("Unsupported clustering algorithms")

    return clustering, GMM_type