# Clustering Methods: Gaussian-means
# input 'X' is X_reduced or X rows

from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_Gmeans(data, X):
    # Number of clusters (can be tuned)
    n_clusters = 2

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # G-means Clustering (using GaussianMixture)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        gmm.fit(X)
        cluster_labels = gmm.predict(X)
        data['cluster'] = cluster_labels
        update_pbar(len(data))

    return