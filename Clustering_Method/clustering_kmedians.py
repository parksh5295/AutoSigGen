# Input data is 'X'; Hstack processing on feature_list
# input 'X' is X_reduced or X rows

from sklearn_extra.cluster import KMedoids
from utils.progressing_bar import progress_bar


def clustering_kmedians(data, state, X):
    kmedians = KMedoids(n_clusters=2, random_state=state)   # default; randomm_state=42

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        data['cluster'] = kmedians.fit_predict(X)

    return