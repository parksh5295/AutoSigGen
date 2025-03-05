# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows

from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar


def clustering_SGMM(data, state, X):
    gmm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=state)   # default; randomm_state=42
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = gmm.fit_predict(X)
        update_pbar(len(data))

    return