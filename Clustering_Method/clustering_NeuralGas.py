# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.gng_replacement import NeuralGasWithParamsSimple   # replaced Neupy

# Replacing Neupy-based function with custom implementation
def clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age):
    model = NeuralGasWithParamsSimple(
        n_start_nodes=n_start_nodes,
        max_nodes=max_nodes,
        step=step,
        max_edge_age=max_edge_age
    )
    model.fit(X)
    clusters = model.labels_
    num_clusters = len(np.unique(clusters))
    return clusters, num_clusters


def clustering_NeuralGas(data, X):
    tune_parameters = Grid_search_all(X, 'NeuralGas')
    print('tune_params: ', tune_parameters)

    best_params = tune_parameters['NeuralGas']['best_params']
    parameter_dict = tune_parameters['NeuralGas']['all_params']
    parameter_dict.update(best_params)

    clusters, num_clusters = clustering_NeuralGas_clustering(
        data, X,
        n_start_nodes=parameter_dict['n_start_nodes'],
        max_nodes=parameter_dict['max_nodes'],
        step=parameter_dict['step'],
        max_edge_age=parameter_dict['max_edge_age']
    )

    data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

    return {
        'Cluster_labeling': data['cluster'],
        'Best_parameter_dict': parameter_dict
    }


# For Grid Search compatibility – use the simple class
class NeuralGasWithParams(BaseEstimator, ClusterMixin):
    def __init__(self, n_start_nodes=2, max_nodes=50, step=0.2, max_edge_age=50):
        self.n_start_nodes = n_start_nodes
        self.max_nodes = max_nodes
        self.step = step
        self.max_edge_age = max_edge_age
        self.model = None
        self.clusters = None

    def fit(self, X, y=None):
        self.model = NeuralGasWithParamsSimple(
            n_start_nodes=self.n_start_nodes,
            max_nodes=self.max_nodes,
            step=self.step,
            max_edge_age=self.max_edge_age
        )
        self.model.fit(X)
        self.clusters = self.model.labels_
        self.labels_ = self.clusters
        return self

    def predict(self, X):
        if self.clusters is None:
            raise RuntimeError("Model must be fit before calling predict()")
        return self.clusters


def pre_clustering_NeuralGas(data, X, n_start_nodes, max_nodes, step, max_edge_age):
    clusters, num_clusters = clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age)
    clustering_data = clustering_nomal_identify(data, clusters, num_clusters)

    return {
        'Cluster_labeling': clustering_data,
        'n_clusters': num_clusters,
        'before_labeling': clusters
    }
