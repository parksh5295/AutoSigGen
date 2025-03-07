# Modules for determining how to cluster

from Clustering_Method.clustering_kmeans import clustering_kmeans
from Clustering_Method.clustering_kmedians import clustering_kmedians
from Clustering_Method.clustering_GMM import clustering_GMM
from Clustering_Method.clustering_SGMM import clustering_SGMM
from Clustering_Method.clustering_Gmeans import clustering_Gmeans
from Clustering_Method.clustering_Xmeans import clustering_Xmeans
from Clustering_Method.clustering_DBSCAN import clustering_DBSCAN
from Clustering_Method.clustering_Mshift import clustering_MShift
from Clustering_Method.clustering_FCM import clustering_FCM
from Clustering_Method.clustering_CK import clustering_CK
from Clustering_Method.clustering_NeuralGas import clustering_NeuralGas


def choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict):
    '''
    Need parameter_dict = {'random_state' : random_state, 'n_init' : n_init, 'max_clusters' : max_clusters, 'eps' : eps,
        'count_samples' : count_samples, 'quantile' : quantile, 'n_samples' : n_samples, 'n_start_nodes' : n_start_nodes,
        'max_nodes' : max_nodes, 'step' : step, 'max_edge_age' : max_edge_age
    }
    '''

    if clustering_algorithm == 'Kmeans':
        clustering = clustering_kmeans(data, X, n_clusters, random_state=parameter_dict['random_state'], n_init=parameter_dict['n_init'])

    elif clustering_algorithm == 'Kmedians':
        clustering = clustering_kmedians(data, X, n_clusters, random_state=parameter_dict['random_state'])

    elif clustering_algorithm == 'GMM':
        GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ")
        clustering = clustering_GMM(data, X, n_clusters, random_state=parameter_dict['random_state'], GMM_type=GMM_type)

    elif clustering_algorithm == 'SGMM':
        clustering = clustering_SGMM(data, X, n_clusters, random_state=parameter_dict['random_state'])

    elif clustering_algorithm == 'Gmeans':
        clustering = clustering_Gmeans(data, X, random_state=parameter_dict['random_state'])

    elif clustering_algorithm == 'Xmeans':
        clustering = clustering_Xmeans(data, X, random_state=parameter_dict['random_state'], max_clusters=parameter_dict['max_clusters'])

    elif clustering_algorithm == 'DBSCAN':
        clustering = clustering_DBSCAN(data, X, eps=parameter_dict['eps'], count_samples=parameter_dict['count_samples'])

    elif clustering_algorithm == 'MShift':
        clustering = clustering_MShift(data, X, random_state=parameter_dict['random_state'], quantile=parameter_dict['quantile'], n_samples=parameter_dict['n_samples'])

    elif clustering_algorithm == 'FCM':
        clustering = clustering_FCM(data, X, n_clusters)

    elif clustering_algorithm == 'CK':
        clustering = clustering_CK(data, X, n_clusters)

    elif clustering_algorithm == 'NeuralGas':
        clustering = clustering_NeuralGas(data, X, n_start_nodes=parameter_dict['n_start_nodes'], max_nodes=parameter_dict['max_nodes'], step=parameter_dict['step'], max_edge_age=parameter_dict['max_edge_age'])

    return clustering