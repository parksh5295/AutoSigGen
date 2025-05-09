# Clustering Algorithm: Custafson-Kessel (Similarly to Fuzzy Algorithm)
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from sklearn.metrics import silhouette_score


# Gustafson-Kessel Clustering Implementation
def ck_cluster(X, c, m=2, error=0.01, maxiter=500, epsilon_scale=1e-5): # Fix. Origin; error=0.005, maxiter=1000
    """
    Gustafson-Kessel Clustering Algorithm.
    
    Parameters:
        X: ndarray
            Input data of shape (n_samples, n_features).
        c: int
            Number of clusters.
        m: float
            Fuzziness coefficient (default=2).
        error: float
            Stopping criterion threshold (default=0.005).
        maxiter: int
            Maximum number of iterations (default=1000).
            
    Returns:
        cntr: ndarray
            Cluster centers of shape (c, n_features).
        u: ndarray
            Final membership matrix of shape (c, n_samples).
        d: ndarray
            Distance matrix of shape (c, n_samples).
        fpc: float
            Final fuzzy partition coefficient.
    """
    n_samples, n_features = X.shape
    u = np.random.dirichlet(np.ones(c), size=n_samples).T  # Random initialization of membership matrix

    um = u ** m

    denom = np.sum(um, axis=1, keepdims=True)
    denom = np.fmax(denom, np.finfo(np.float64).eps)  # Prevent 0
    cntr = np.dot(um, X) / denom

    cov_matrices = np.array([np.eye(n_features) for _ in range(c)])  # Initial covariance matrices
    d = np.zeros((c, n_samples))

    for iteration in range(maxiter):
        # Calculate cluster centers
        cntr = np.dot(um, X) / um.sum(axis=1, keepdims=True)

        # Update covariance matrices
        for i in range(c):
            diff = X - cntr[i]
            cov = np.dot((um[i][:, np.newaxis] * diff).T, diff) / um[i].sum()

            # Normalize by determinant
            det = np.linalg.det(cov)
            if not np.isfinite(det) or det <= 0:    # Exception handling
                det = np.finfo(float).eps
            cov /= det ** (1 / n_features)

            # Regularize covariance
            cov = regularize_covariance(cov, epsilon_scale)

            cov_matrices[i] = cov
        '''
        # Checking matrix dimensions
        print("um[i].shape:", um[i].shape)
        print("diff.shape:", diff.shape)
        print("cov_matrices[i].shape:", cov_matrices[i].shape)
        '''
        
        # Calculate distances and update membership
        for i in range(c):
            diff = X - cntr[i]
            
            try:
                inv_cov = np.linalg.inv(cov_matrices[i])
            except np.linalg.LinAlgError:
                print(f"[WARNING] Singular matrix at cluster {i}, using pseudo-inverse.")
                inv_cov = np.linalg.pinv(cov_matrices[i])

            val = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            val = np.clip(val, 0, None) # Prevent negative numbers before SQRT
            d[i] = np.sqrt(val)

        d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid division by zero
        
        ratio = d / d[:, np.newaxis]
        ratio = np.clip(ratio, np.finfo(np.float64).eps, 1e10)  # Preventing very small to too large numbers
        u_new = 1.0 / np.sum(ratio ** (2 / (m - 1)), axis=0)

        # Check for convergence
        if np.linalg.norm(u_new - u) < error:
            break
        u = u_new

    fpc = np.sum(u ** m) / n_samples
    return cntr, u, d, fpc, cov_matrices


def ck_predict(X_new, cntr, cov_matrices, m=2):
    """
    Predict membership for new data points in Gustafson-Kessel clustering.

    Parameters:
        X_new: ndarray
            New data of shape (n_samples, n_features).
        cntr: ndarray
            Cluster centers of shape (c, n_features).
        cov_matrices: ndarray
            Covariance matrices of shape (c, n_features, n_features).
        m: float
            Fuzziness coefficient.

    Returns:
        membership: ndarray
            Membership matrix of shape (c, n_samples).
    """
    c = cntr.shape[0]
    n_samples = X_new.shape[0]
    d = np.zeros((c, n_samples))

    for i in range(c):
        diff = X_new - cntr[i]
        inv_cov = np.linalg.inv(cov_matrices[i])
        d[i] = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))

    d = np.fmax(d, np.finfo(np.float64).eps)  # Avoid divide-by-zero
    u = 1.0 / np.sum((d / d[:, np.newaxis]) ** (2 / (m - 1)), axis=0)

    return u


def clustering_CK(data, X, max_clusters):
    after_elbow = Elbow_method(data, X, 'CK', max_clusters)
    n_clusters = after_elbow['optimul_cluster_n']
    parameter_dict = after_elbow['parameter_dict']

    # Perform Gustafson-Kessel Clustering; Performing with auto-tuned epsilon included
    cntr, u, d, fpc, cov_matrices, best_epsilon = tune_epsilon_for_ck(X, c=n_clusters)
    parameter_dict['epsilon_scale'] = best_epsilon  # Save selected values

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    # Debug cluster id
    print(f"\n[DEBUG CK main_clustering] Param for CNI 'data' - Shape: {data.shape}")
    print(f"[DEBUG CK main_clustering] Param for CNI 'data' - Columns: {list(data.columns)}")
    
    print(f"[DEBUG CK main_clustering] Array used for clustering 'X' - Shape: {X.shape}")
    # if not hasattr(X, 'columns'):
    #     print(f"[DEBUG CK main_clustering] Array used for clustering 'X' (NumPy array) - First 5 cols of first row: {X[0, :5] if X.shape[0] > 0 and X.shape[1] >= 5 else 'N/A or too small'}")
    
    data['cluster'] = clustering_nomal_identify(data, cluster_labels, n_clusters)

    predict_CK = data['cluster']

    return {
        'Cluster_labeling': predict_CK,
        'Best_parameter_dict': parameter_dict
    }


def pre_clustering_CK(data, X, n_clusters):
    # Perform Gustafson-Kessel Clustering
    cntr, u, d, fpc, cov_matrices = ck_cluster(X, c=n_clusters, m=2)

    # Assign clusters based on maximum membership
    cluster_labels = np.argmax(u, axis=0)

    predict_CK = clustering_nomal_identify(data, cluster_labels, n_clusters)
    num_clusters = len(np.unique(predict_CK))  # Counting the number of clusters

    # Wrapping to write like a model
    ck_model = CKFakeModel(cntr, cov_matrices, fpc)

    return {
        'Cluster_labeling' : predict_CK,
        'n_clusters' : num_clusters,
        'before_labeling' : ck_model
    }

# Make it look like a model for Elbow/tuning
class CKFakeModel:
    def __init__(self, cntr, cov_matrices, fpc):
        self.cntr = cntr
        self.cov_matrices = cov_matrices
        self.fpc = fpc
        self.inertia_ = 1 - fpc  # Just for Elbow_method compatibility

    def predict(self, X_new):
        u = ck_predict(X_new, self.cntr, self.cov_matrices)
        return np.argmax(u, axis=0)

    def fit(self, X):
        pass  # Already fitted


# Functions for stabilizing CK covariance
def regularize_covariance(cov, epsilon_scale=1e-5):
    """
    Regularize covariance matrix if it is near-singular.
    
    Parameters:
        cov: ndarray
            Covariance matrix.
        epsilon_scale: float
            Scale of epsilon added to the diagonal, relative to mean of diagonal.

    Returns:
        cov_reg: ndarray
            Regularized covariance matrix.
    """
    try:
        # Check determinant for near-singularity
        cond = np.linalg.cond(cov)
        if cond > 1e10 or np.isnan(cond):
            avg_diag = np.mean(np.diag(cov))
            if avg_diag == 0 or np.isnan(avg_diag):
                avg_diag = 1.0
            epsilon = epsilon_scale * avg_diag
            cov += np.eye(cov.shape[0]) * epsilon
    except np.linalg.LinAlgError:
        # If condition number fails
        cov += np.eye(cov.shape[0]) * epsilon_scale
    return cov

def tune_epsilon_for_ck(X, c, epsilon_candidates=[1e-7, 1e-6, 1e-5, 1e-4]):
    best_score = -1
    best_result = None
    best_epsilon = None

    for eps in epsilon_candidates:
        try:
            cntr, u, d, fpc, cov_matrices = ck_cluster(X, c=c, m=2, epsilon_scale=eps)
            labels = np.argmax(u, axis=0)

            score = silhouette_score(X, labels) # Or other metrics, such as FPC

            if score > best_score:
                best_score = score
                best_result = (cntr, u, d, fpc, cov_matrices)
                best_epsilon = eps

        except np.linalg.LinAlgError:
            continue    # If we get a singular matrix error, ignore it and move to the next epsilon with the

    return best_result + (best_epsilon,)
