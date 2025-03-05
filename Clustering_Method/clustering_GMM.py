# input 'X' is X_reduced or X rows

from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_GMM_normal(data, state, X):
    gmm = GaussianMixture(n_components=2, random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))
    return

def clustering_GMM_full(data, state, X):
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))
    return

def clustering_GMM_tied(data, state, X):
    gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))
    return

def clustering_GMM_diag(data, state, X):
    gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))
    return


def clustering_GMM(data, state, X, GMM_type):
    if GMM_type == 'normal':
        clustering_GMM_normal(data, state, X)
    elif GMM_type == 'full':
        clustering_GMM_full(data, state, X)
    elif GMM_type == 'tied':
        clustering_GMM_tied(data, state, X)
    elif GMM_type == 'diag':
        clustering_GMM_diag(data, state, X)
    else:
        print("GMM type Error!! -In Clustering")
    
    return