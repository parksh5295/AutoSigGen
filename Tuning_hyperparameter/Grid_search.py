# Input and output is parameter_dict
# The GridSearchCV model is manually implemented: Evaluate based on silhouette score
'''
Output: Dictionaries by Clustering algorithm
xmeans_result = best_results['Xmeans']  ->  {'best_params': {'max_clusters': 50}, 'all_params': {parameter_dict}, 'silhouette_score': 0.78, 'davies_bouldin_score': 0.42}
best_xmeans_params = best_results['Xmeans']['best_params']  ->  {'max_clusters': 50}
'''

import numpy as np
import importlib
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score, f1_score, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from utils.class_row import nomal_class_data
from Clustering_Method.common_clustering import get_clustering_function


# Dynamic import functions (using importlib)
def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def Grid_search_Kmeans(X, n_clusters, parameter_dict=None):
    # Maintain complete parameter_dict for compatibility
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5
        }

    n_init_values = list(range(2, 102, 3))
    best_score = -1
    best_params = None

    for n_init in n_init_values:
        # Use only necessary parameters for KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=parameter_dict['random_state'],
            n_init=n_init
        )
        labels = kmeans.fit_predict(X)

        # Make sure Silhouette score is calculable
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = {'n_init': n_init}

    # Merge with complete parameter_dict
    best_param_full = parameter_dict.copy()
    if best_params:
        best_param_full.update(best_params)

    return best_param_full


'''
Grid Search functions for clustering algorithms other than Kmeans
'''

def evaluate_clustering(X, labels):
    """Functions to evaluate clustering performance (Silhouette Score & Davies-Bouldin Score)"""
    if len(set(labels)) < 2:
        return -1, float('inf')  # Returning an invalid score if there is only one cluster
    
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return sil_score, db_score


# Discriminate Functions for CANNwKNN
def evaluate_clustering_with_known_benign(data, X, clusters, num_clusters, aligned_original_labels):
    # 0: benign, 1: attack (criteria: same as clustering_nomal_identify)
    benign_data = nomal_class_data(data).to_numpy() # Assuming that we only know benign data

    inferred_labels = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    # 실Create ground truth about where my benigns are
    ground_truth = np.ones(len(data))  # The default is attack(1)
    benign_idx = data.index.isin(benign_data.index)
    ground_truth[benign_idx] = 0

    f1 = f1_score(ground_truth, inferred_labels)
    acc = accuracy_score(ground_truth, inferred_labels)
    return f1, acc


def Grid_search_all(X, clustering_algorithm, parameter_dict=None, data=None):
    print("[DEBUG Grid_search_all] Received X for grid search. Shape:", X.shape)
    if hasattr(X, 'dtypes'):
        print("[DEBUG Grid_search_all] Dtypes of X:\n", X.dtypes)
    elif hasattr(X, 'dtype'):
        print("[DEBUG Grid_search_all] Dtype of X (NumPy array):", X.dtype)
    if hasattr(X, 'head'):
        print("[DEBUG Grid_search_all] Head of X (DataFrame):\n", X.head(3))
    elif isinstance(X, np.ndarray):
        print("[DEBUG Grid_search_all] First 2 rows of X (NumPy array):\n", X[:2])

    # Maintain complete parameter_dict for compatibility
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5
        }

    best_results = {}  # Dictionary for storing the best result of each algorithm

    print(f"\n{clustering_algorithm} Performing clustering...")

    if clustering_algorithm in ['Xmeans', 'xmeans']:
        XMeansWrapper = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
        param_grid = {'max_clusters': list(range(2, 31, 3))}
        def create_model(params):
            # Use only necessary parameters for XMeans
            model_params = {
                'random_state': parameter_dict['random_state'],
                **params
            }
            return XMeansWrapper(**model_params)

    elif clustering_algorithm in ['Gmeans', 'gmeans']:
        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
        log_range = np.logspace(-6, -1, num=10)
        lin_range = np.linspace(min(log_range), max(log_range), num=10)
        combined_range = np.unique(np.concatenate((log_range, lin_range)))

        param_grid = {'max_clusters': list(range(2, 31, 3)), 'tol': combined_range}
        def create_model(params):
            # Use only necessary parameters for GMeans
            model_params = {
                'random_state': parameter_dict['random_state'],
                **params
            }
            return GMeans(**model_params)

    elif clustering_algorithm == 'DBSCAN':
        param_grid = {'eps': np.arange(0.1, 1, 0.02), 'min_samples': list(range(3, 20, 2))}
        def create_model(params):
            return DBSCAN(**params)

    elif clustering_algorithm == 'MShift':
        MeanShiftWithDynamicBandwidth = dynamic_import("Clustering_Method.clustering_Mshift", "MeanShiftWithDynamicBandwidth")
        param_grid = {'quantile': np.arange(0.01, 0.31, 0.05), 'n_samples': list(range(50, 210, 30))}    # Bandwidth estimates can be erroneous if n_samples is too large(1000)
        def create_model(params):
            return MeanShiftWithDynamicBandwidth(**params)

    elif clustering_algorithm == 'NeuralGas':
        NeuralGasWithParams = dynamic_import("Clustering_Method.clustering_NeuralGas", "NeuralGasWithParams")
        '''
        param_grid = {'n_start_nodes': [2, 5, 7, 10, 15, 20, 35, 50], 'max_nodes': list(range(50, 501, 50)),
                        'step': np.arange(0.05, 0.51, 0.05), 'max_edge_age': list(range(50, 301, 30))}
        def create_model(params):
            return NeuralGasWithParams(**params)
        '''
        # Automatically calculate reasonable ranges based on data counts
        n = len(X)
        estimated_nodes = int(np.sqrt(n))  # Recommended default values

        # Limiting the max_nodes range
        max_nodes_list = [int(0.5 * estimated_nodes), estimated_nodes, int(1.5 * estimated_nodes)]
        max_nodes_list = [m for m in max_nodes_list if m >= 10]  # Exclude values that are too small

        # Create constrained combinations to make max_edge_age proportional to max_nodes
        edge_age_list = lambda m: [int(0.5 * m), m, int(1.5 * m)]

        param_combinations = []
        for start_nodes in [2, 5, 10]:
            for max_nodes in max_nodes_list:
                for edge_age in edge_age_list(max_nodes):
                    for step in [0.1, 0.2, 0.3]:
                        param_combinations.append({
                            'n_start_nodes': start_nodes,
                            'max_nodes': max_nodes,
                            'step': step,
                            'max_edge_age': edge_age
                        })

        def create_model(params):
            return NeuralGasWithParams(**params)

    elif clustering_algorithm in ['CANNwKNN', 'CANN']:
        CANNWithKNN = dynamic_import("Clustering_Method.clustering_CANNwKNN", "CANNWithKNN")
        param_grid = {'epochs': list(range(20, 501, 20)), 'batch_size': list(range(32, 257, 32)), 
                        'n_neighbors': list(range(3, 51, 5))}
        input_shape = X.shape[1]
        def create_model(params):
            # Use only necessary parameters for CANNwKNN
            model_params = {
                'input_shape': input_shape,
                **params
            }
            return CANNWithKNN(**model_params)

    else:
        print(f"Unsupported algorithm: {clustering_algorithm}")
        pass

    if clustering_algorithm == 'NeuralGas':
        # ... param_combinations already created (see above)
        pass
    else:
        # Generate all hyperparameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

    best_score = -1
    best_db_score = float('inf')
    best_params = None

    for param_set in param_combinations:
        if clustering_algorithm == 'NeuralGas':
            params = param_set  # Already a dict
        else:
            params = dict(zip(param_keys, param_set))
        model = create_model(params)

        print("[DEBUG] Grid_search_all() - X for training:", X.shape)
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            # Debugging
            X_first_row_list = X.iloc[0].tolist()
            data_first_row_list = X.iloc[0].tolist()
            print(X_first_row_list)
            print(data_first_row_list)
            data_only = [item for item in data_first_row_list if item not in X_first_row_list]
            print("in data only: ", data_only)
            # End of Debugging
            labels = model.fit_predict(X, data)
        else:
            labels = model.fit_predict(X)
        sil_score, db_score = evaluate_clustering(X, labels)

        print(f"{clustering_algorithm}: {params} → Silhouette: {sil_score:.4f}, Davies-Bouldin: {db_score:.4f}")

        # Select if you have a high Silhouette Score and a low Davies-Bouldin Score
        if sil_score > best_score or (sil_score == best_score and db_score < best_db_score):
            best_score = sil_score
            best_db_score = db_score
            best_params = params

    best_results[clustering_algorithm] = {
        'best_params': best_params,
        'all_params': parameter_dict,  # Return complete parameter_dict for compatibility
        'silhouette_score': best_score,
        'davies_bouldin_score': best_db_score
    }

    return best_results