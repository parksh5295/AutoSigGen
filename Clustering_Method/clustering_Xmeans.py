# input 'X' is X_reduced or X rows
# Clustering Algorithm: X-means; Autonomously tuning n_clusters in k-means
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify

# X-Means Clustering Function
def x_means_clustering(X, random_state, max_clusters):
    best_score = -1
    best_model = None
    best_k = 2
    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=50)
        labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_model = model
            best_k = k
    return best_model, best_k

def clustering_Xmeans_clustering(data, X, random_state, max_clusters):  # Fundamental Xmeans clustering
    # default; max=clusters=10,
    # Perform Y-Means Clustering
    model, optimal_k = x_means_clustering(X, random_state, max_clusters)
    clusters = model.labels_

    return clusters, optimal_k


def clustering_Xmeans(data, X, aligned_original_labels):
    tune_parameters = Grid_search_all(X, 'Xmeans')
    best_params = tune_parameters['Xmeans']['best_params']
    parameter_dict = tune_parameters['Xmeans']['all_params']
    parameter_dict.update(best_params)

    clusters, num_clusters = clustering_Xmeans_clustering(data, X, random_state=parameter_dict['random_state'], max_clusters=parameter_dict['max_clusters'])

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG XMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG XMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters)

    # predict_Xmeans = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict
    }


# Auxiliary class for Grid Search
class XMeansWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, random_state=42, max_clusters=10):
        # Automatically assign a value for __init__ if no input value is present
        self.random_state = random_state
        self.max_clusters = max_clusters
        self.model = None
        self.best_k = None

    def fit(self, X, y=None):
        self.model, self.best_k = x_means_clustering(X, self.random_state, self.max_clusters)
        return self

    def predict(self, X):
        return self.model.predict(X) if self.model else None

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
    

def pre_clustering_Xmeans(data, X, random_state, max_clusters):
    # clusters are model-generated labels before CNI, num_clusters_optimal is the k found by x_means_clustering
    cluster_labels, num_clusters_optimal = clustering_Xmeans_clustering(data, X, random_state, max_clusters)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_optimal, # Optimal n_clusters from XMeans
        'before_labeling' : cluster_labels # or the model from x_means_clustering if needed, but usually labels are sufficient for pre_clustering
    }