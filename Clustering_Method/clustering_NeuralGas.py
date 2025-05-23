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


def clustering_NeuralGas(data, X, aligned_original_labels):
    print("[DEBUG clustering_NeuralGas] Received X for clustering. Shape:", X.shape)
    if hasattr(X, 'dtypes'):
        print("[DEBUG clustering_NeuralGas] Dtypes of X:\n", X.dtypes)
    elif hasattr(X, 'dtype'):
        print("[DEBUG clustering_NeuralGas] Dtype of X (NumPy array):", X.dtype)
    if hasattr(X, 'head'):
        print("[DEBUG clustering_NeuralGas] Head of X (DataFrame):\n", X.head(3))
    elif isinstance(X, np.ndarray):
        print("[DEBUG clustering_NeuralGas] First 2 rows of X (NumPy array):\n", X[:2])

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

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG NeuralGas main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG NeuralGas main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
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
    # cluster_labels are model-generated labels, num_clusters_actual is the count of unique labels found by NeuralGas
    cluster_labels, num_clusters_actual = clustering_NeuralGas_clustering(data, X, n_start_nodes, max_nodes, step, max_edge_age)
    
    # predict_NeuralGas = clustering_nomal_identify(data, cluster_labels, num_clusters_actual)
    # num_clusters = len(np.unique(predict_NeuralGas))  # Counting the number of clusters

    # For NeuralGas, the 'before_labeling' might be the model itself if its state is useful, or just cluster_labels.
    # Here, returning cluster_labels for consistency with other pre_clustering functions that return labels or simple model objects.
    # If the NeuralGasWithParamsSimple model object is needed by tuning methods, this might need adjustment.
    neural_gas_model_placeholder = cluster_labels # Or potentially the model from clustering_NeuralGas_clustering if it's serializable and useful

    return {
        'model_labels' : cluster_labels,
        'n_clusters': num_clusters_actual, # Actual n_clusters from NeuralGas
        'before_labeling': neural_gas_model_placeholder 
    }
