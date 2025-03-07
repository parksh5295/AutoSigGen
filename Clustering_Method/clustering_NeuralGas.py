# input 'X' is X_reduced or X rows

import numpy as np
from neupy.algorithms import GrowingNeuralGas
from utils.progressing_bar import progress_bar
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify


def clustering_NeuralGas_clustering(data, n_start_nodes, max_nodes, step, max_edge_age, X):  # Fundamental NeuralGas clustering
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


def clustering_NeuralGas(data, X, n_start_nodes, max_nodes, step, max_edge_age):
    clusters, num_clusters = clustering_NeuralGas_clustering(data, n_start_nodes, max_nodes, step, max_edge_age, X)
    data['cluster'] = clustering_nomal_identify(data, clusters, num_clusters)

    predict_NeuralGas = data['cluster']

    return predict_NeuralGas