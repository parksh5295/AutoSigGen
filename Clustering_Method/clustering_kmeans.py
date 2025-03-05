# input 'X' is X_reduced or X rows

from sklearn.cluster import KMeans
from utils.progressing_bar import progress_bar


def clustering_kmeans(data, state, init, X):
    # Apply KMeans Clustering
    num_clusters = 2  # Assuming binary classification
    kmeans = KMeans(n_clusters=num_clusters, random_state=state, n_init=init)   # default; randomm_state=42, n_init=10

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = kmeans.fit_predict(X)
        update_pbar(len(data))
    
    return