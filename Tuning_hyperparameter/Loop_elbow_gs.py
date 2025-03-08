# Number of Cluster -> Elbow Method
# Others Hyperparameters -> Grid Search Method
# Loop; using Elbow method and Grid Search method as threads

# Some Clustering Algorihtm; Kmeans requires this LOOP Hyperparameter TUNING system.


from Tuning_hyperparameter.Elbow_method import Elbow_method


def loop_tuning(data, X, clustering_algorithm):
    first_parameter_dict = {'random_state' : 42, 'n_init' : 30, 'max_clusters' : 50, 'eps' : 0.5, 'count_samples' : 5, 'quantile' : 0.2,
                            'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2, 'max_edge_age' : 50,
                            }
    max_clusters = 10000

    before_n_cluster = Elbow_method(data, X, clustering_algorithm, max_clusters, first_parameter_dict)

    while not 0.99 < before_n_cluster/after_n_cluster < 1.01:

    
