# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from neupy.algorithms import GrowingNeuralGas
from sklearn.base import BaseEstimator, ClusterMixin
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age):  # Fundamental NeuralGas clustering
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Apply Neural Gas clustering
        neural_gas = GrowingNeuralGas(n_inputs = X.shape[1], n_start_nodes=n_start_nodes, max_nodes=max_nodes, step=step, max_edge_age=max_edge_age)
        # default; n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50
        neural_gas.train(X, epochs=100)

        # Find the closest node for each data point
        def assign_cluster(data_point, graph):
            distances = [np.linalg.norm(data_point - node.weight) for node in graph.nodes]
            return np.argmin(distances)
        
        # Assign clusters to data points
        clusters = np.array([assign_cluster(x, neural_gas.graph) for x in X])

        update_pbar(len(data))

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters

    return clusters, num_clusters


def clustering_NeuralGas(data, X):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        tune_parameters = Grid_search_all(X, 'NeuralGas')
        best_params = tune_parameters['NeuralGas']['best_params']
        parameter_dict = tune_parameters['NeuralGas']['all_params']
        parameter_dict.update(best_params)

        clusters, num_clusters = clustering_NeuralGas_clustering(data, X, n_start_nodes=parameter_dict['n_start_nodes'], max_nodes=parameter_dict['max_nodes'], step=parameter_dict['step'], max_edge_age=parameter_dict['max_edge_age'])
        data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

        update_pbar(len(data))

    predict_NeuralGas = data['cluster']

    return {
        'Cluster_labeling': predict_NeuralGas,
        'Best_parameter_dict': parameter_dict
    }


# Additional classes for Grid Search
class NeuralGasWithParams(BaseEstimator, ClusterMixin):
    def __init__(self, n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50):
        self.n_start_nodes = n_start_nodes
        self.max_nodes = max_nodes
        self.step = step
        self.max_edge_age = max_edge_age
        self.model = None
        self.clusters = None

    def fit(self, X, y=None):
        # NeuralGas 클러스터링 적용
        neural_gas = GrowingNeuralGas(n_inputs=X.shape[1], n_start_nodes=self.n_start_nodes, 
                                      max_nodes=self.max_nodes, step=self.step, max_edge_age=self.max_edge_age)
        neural_gas.train(X, epochs=100)

        # 데이터 포인트에 대해 클러스터 할당
        def assign_cluster(data_point, graph):
            distances = [np.linalg.norm(data_point - node.weight) for node in graph.nodes]
            return np.argmin(distances)

        self.clusters = np.array([assign_cluster(x, neural_gas.graph) for x in X])
        return self
    
    def predict(self, X):
        return self.clusters