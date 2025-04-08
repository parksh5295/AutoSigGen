# For PCA before main clustering
# Input data is usally X; after that Hstack processing
# Output data is usally 'X_reduced'

from sklearn.decomposition import PCA


def pca_func(data, number_of_components=10, state=42):  # default; n_components=10, state=42
    pca = PCA(n_components=number_of_components, random_state=state)    # default; n_components=10
    data_reduced = pca.fit_transform(data)
    return data_reduced