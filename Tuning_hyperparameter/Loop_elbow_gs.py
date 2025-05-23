# Number of Cluster -> Elbow Method
# Others Hyperparameters -> Grid Search Method
# Loop; using Elbow method and Grid Search method as threads
# Output is Dictionary; 'optimul_cluster_n'(int) and 'best_parameter_dict'(dict)

# Some Clustering Algorihtm; Kmeans requires this LOOP Hyperparameter TUNING system.


from Tuning_hyperparameter.Elbow_method import Elbow_method
from Tuning_hyperparameter.Grid_search import Grid_search_Kmeans


def loop_tuning(data, X, clustering_algorithm, max_clusters=10000):
    # Maintain complete parameter_dict for compatibility
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
    # First_parameter_dictionary

    elbow_result = Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict)
    before_n_cluster = elbow_result['optimul_cluster_n']

    new_parameter_dict = parameter_dict # Initialize new_parameter_dict

    # Loop until cluster number stabilizes
    for _ in range(1000):  # Safety limit
        new_params = Grid_search_Kmeans(X, before_n_cluster, parameter_dict)
        elbow_result = Elbow_method(data, X, clustering_algorithm, max_clusters, new_params)
        after_n_cluster = elbow_result['optimul_cluster_n']

        if 0.99 < before_n_cluster / after_n_cluster < 1.01:
            break
        before_n_cluster = after_n_cluster
        new_parameter_dict = new_params

    return {
        'optimul_cluster_n': after_n_cluster,
        'best_parameter_dict': new_parameter_dict  # Return complete parameter_dict for compatibility
    }