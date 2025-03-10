# Input and output is parameter_dict
# The GridSearchCV model is manually implemented: Evaluate based on silhouette score
'''
Output: Dictionaries by Clustering algorithm
xmeans_result = best_results['Xmeans']  ->  {'best_params': {'max_clusters': 50}, 'all_params': {parameter_dict}, 'silhouette_score': 0.78, 'davies_bouldin_score': 0.42}
best_xmeans_params = best_results['Xmeans']['best_params']  ->  {'max_clusters': 50}
'''

import numpy as np
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score

from sklearn.cluster import KMeans
from Clustering_Method.clustering_Xmeans import XMeansWrapper
from Clustering_Method.clustering_Gmeans import GMeans
from sklearn.cluster import DBSCAN
from Clustering_Method.clustering_Mshift import MeanShiftWithDynamicBandwidth
from Clustering_Method.clustering_NeuralGas import NeuralGasWithParams
from Clustering_Method.clustering_CANNwKNN import CANNWithKNN


def Grid_search_Kmeans(X, n_clusters, parameter_dict=None):
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42, 'n_init': 30, 'max_clusters': 1000, 'tol': 1e-4, 'eps': 0.5, 'count_samples': 5, 'quantile': 0.2, 
            'n_samples': 500,'n_start_nodes': 2, 'max_nodes': 50, 'step': 0.2, 'max_edge_age': 50, 'epochs': 300,
            'batch_size': 256, 'n_neighbors': 5
        }

    silhouette_scorer = make_scorer(silhouette_score, greater_is_better=True)

    param_grid = {
        'n_init': list(range(2, 102, 3))
    }
    clustering_al = KMeans(random_state=parameter_dict['random_state'], n_clusters=n_clusters)

    grid_search = GridSearchCV(clustering_al, param_grid, scoring=silhouette_scorer, cv=5)
    grid_search.fit(X)

    Best_parameter_dict = grid_search.best_params

    return Best_parameter_dict


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


def Grid_search_all(X, algorithms, parameter_dict=None):
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42, 'n_init': 30, 'max_clusters': 1000, 'tol': 1e-4, 'eps': 0.5, 'count_samples': 5, 'quantile': 0.2, 
            'n_samples': 500, 'n_start_nodes': 2, 'max_nodes': 50, 'step': 0.2, 'max_edge_age': 50, 'epochs': 300,
            'batch_size': 256, 'n_neighbors': 5
        }

    best_results = {}  # Dictionary for storing the best result of each algorithm

    for clustering_algorithm in algorithms:
        print(f"\n{clustering_algorithm} Performing clustering...")

        if clustering_algorithm == 'Xmeans':
            param_grid = {
                'max_clusters': list(range(10, 101, 10))
                }
            def create_model(params):
                return XMeansWrapper(random_state=parameter_dict['random_state'], **params)

        elif clustering_algorithm == 'Gmeans':
            log_range = np.logspace(-6, -1, num=10)
            lin_range = np.linspace(min(log_range), max(log_range), num=10)
            combined_range = np.unique(np.concatenate((log_range, lin_range)))

            param_grid = {
                'max_clusters': list(range(10, 1001, 10)),
                'tol': combined_range
                }
            def create_model(params):
                return GMeans(random_state=parameter_dict['random_state'], **params)

        elif clustering_algorithm == 'DBSCAN':
            param_grid = {
                'eps': np.arange(0.05, 1, 0.02),
                'min_samples': list(range(3, 20, 2))
                }
            def create_model(params):
                return DBSCAN(**params)

        elif clustering_algorithm == 'MShift':
            param_grid = {
                'quantile': np.arange(0.01, 0.95, 0.02),
                'n_samples': list(range(50, 1000, 50))
                }
            def create_model(params):
                return MeanShiftWithDynamicBandwidth(**params)

        elif clustering_algorithm == 'NeuralGas':
            param_grid = {'n_start_nodes': list(range(1, 11, 1)),
                          'max_nodes': list(range(10, 101, 5)),
                          'step': np.arange(0.05, 1, 0.05),
                          'max_edge_age': list(range(5, 301, 10))
                          }
            def create_model(params):
                return NeuralGasWithParams(**params)

        elif clustering_algorithm == 'CANNwKNN':
            param_grid = {'epochs': list(range(10, 501, 10)),
                          'batch_size': list(range(32, 257, 32)),
                          'n_neighbors': list(range(5, 51, 1))
                          }
            input_shape = (X.shape[1],)
            def create_model(params):
                return CANNWithKNN(input_shape=input_shape, **params)

        else:
            print(f"Unsupported algorithms: {clustering_algorithm}")
            continue

        # Generate all hyperparameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        best_score = -1
        best_db_score = float('inf')
        best_params = None

        for param_set in param_combinations:
            params = dict(zip(param_keys, param_set))
            model = create_model(params)

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
            'all_params' : parameter_dict,
            'silhouette_score': best_score,
            'davies_bouldin_score': best_db_score
        }

    return best_results