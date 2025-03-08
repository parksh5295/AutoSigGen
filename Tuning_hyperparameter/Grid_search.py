from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, silhouette_score

from sklearn.cluster import KMeans


def Grid_search(data, X, clustering_algorithm, n_clusters, parameter_dict):
    silhouette_scorer = make_scorer(silhouette_score, greater_is_better=True)

    if clustering_algorithm == 'Kmeans':
        param_grid = {
            'n_init': list(range(2, 100, 3)),  
            'tol': [1e-4, 1e-5, 1e-6]
        }

        clustering = KMeans(random_state=parameter_dict['random_state'], n_clusters=n_clusters)
    
    grid_search = GridSearchCV(clustering, param_grid, scoring=silhouette_scorer, cv=5)
    grid_search.fit(X)

    Best_parameter_dict = grid_search.best_params

    return Best_parameter_dict


