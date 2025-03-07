# input 'X' is X_reduced or X rows

from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_GMM_normal(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']

    return predict_GMM


def clustering_GMM_full(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']

    return predict_GMM


def clustering_GMM_tied(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']

    return predict_GMM


def clustering_GMM_diag(data, state, X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    predict_GMM = data['cluster']

    return predict_GMM


def clustering_GMM(data, X, n_clusters, state, GMM_type):
    if GMM_type == 'normal':
        predict_GMM = clustering_GMM_normal(data, state, X, n_clusters)
    elif GMM_type == 'full':
        predict_GMM = clustering_GMM_full(data, state, X, n_clusters)
    elif GMM_type == 'tied':
        predict_GMM = clustering_GMM_tied(data, state, X, n_clusters)
    elif GMM_type == 'diag':
        predict_GMM = clustering_GMM_diag(data, state, X, n_clusters)
    else:
        print("GMM type Error!! -In Clustering")
    
    return predict_GMM