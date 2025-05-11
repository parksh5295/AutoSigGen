# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Clustering_Method.common_clustering import get_clustering_function


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict, GMM_type="normal"):   # X: Encoding and embedding, post-PCA, post-delivery
    pre_clustering_func = get_clustering_function(clustering_algorithm)

    # Parameters are expected to be in parameter_dict passed from Elbow_method
    if clustering_algorithm == 'Kmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'n_init': parameter_dict['n_init']
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'GMM': # General GMM, distinct from SGMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'GMM_type': GMM_type, # GMM_type is specific to pre_clustering_GMM
            'reg_covar': parameter_dict['reg_covar']
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'SGMM': # Spherical GMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'reg_covar': parameter_dict['reg_covar']
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm in ['FCM', 'CK']:
        clustering = pre_clustering_func(data, X, n_clusters)
    elif clustering_algorithm == 'Gmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'], # GMeans uses max_clusters from dict
            'tol': parameter_dict['tol']
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Gmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'Xmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'] # XMeans uses max_clusters from dict
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Xmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'DBSCAN':
        algorithm_params = {
            # DBSCAN does not use random_state in its sklearn implementation
            'eps': parameter_dict['eps'],
            'count_samples': parameter_dict['count_samples'] # maps to min_samples
        }
        # n_clusters (k from loop) is not passed to pre_clustering_DBSCAN
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'MShift':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'quantile': parameter_dict['quantile'],
            'n_samples': parameter_dict['n_samples']
        }
        # n_clusters (k from loop) is not passed to pre_clustering_MShift
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'NeuralGas':
        algorithm_params = {
            # NeuralGas pre_clustering doesn't take random_state in its current signature
            'n_start_nodes': parameter_dict['n_start_nodes'],
            'max_nodes': parameter_dict['max_nodes'],
            'step': parameter_dict['step'],
            'max_edge_age': parameter_dict['max_edge_age']
        }
        # n_clusters (k from loop) is not passed to pre_clustering_NeuralGas
        clustering = pre_clustering_func(data, X, **algorithm_params)
    else: # KMedians and any other algorithm that takes n_clusters and random_state by default
        algorithm_params = {
            'random_state': parameter_dict['random_state']
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict=None):
    # Maintain complete parameter_dict for compatibility and to ensure all necessary params are available
    base_parameter_dict = {
        'random_state': 42,
        'n_init': 30, # For KMeans primarily
        'max_clusters': 1000, # For GMeans, XMeans primarily
        'tol': 1e-4, # For GMeans primarily
        'eps': 0.5, # For DBSCAN primarily
        'count_samples': 5, # For DBSCAN primarily (as min_samples)
        'quantile': 0.2, # For MShift primarily
        'n_samples': 500, # For MShift primarily
        'n_start_nodes': 2, # For NeuralGas primarily
        'max_nodes': 50, # For NeuralGas primarily
        'step': 0.2, # For NeuralGas primarily
        'max_edge_age': 50, # For NeuralGas primarily
        'epochs': 300, # For CANNwKNN (if used here, but usually tuned in Grid_search_all)
        'batch_size': 256, # For CANNwKNN
        'n_neighbors': 5, # For CANNwKNN
        'reg_covar': 1e-6 # For GMM and SGMM
    }
    if parameter_dict is not None:
        base_parameter_dict.update(parameter_dict) # Update with any user-provided params
    
    current_parameter_dict = base_parameter_dict.copy()

    wcss_or_bic = []  # Store WCSS (inertia) or BIC by number of clusters
    cluster_range = range(2, max_clusters + 1) # Start from 2 clusters for GMM/SGMM BIC calculation and most algos
    if not cluster_range:
        # Handle edge case where max_clusters < 2
        print("Warning: max_clusters is less than 2. Elbow method requires at least 2 clusters. Returning default k=2.")
        return {
            'optimul_cluster_n': 2,
            'best_parameter_dict': current_parameter_dict
        }

    for k in cluster_range:
        # Pass the current_parameter_dict to Elbow_choose_clustering_algorithm
        # This dict contains reg_covar and other relevant params.
        # Elbow_choose_clustering_algorithm will then extract relevant params for the specific pre_clustering_func.
        clustering_result = Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, k, current_parameter_dict)
        
        # Ensure clustering_result is not None and contains 'before_labeling'
        if clustering_result is None or 'before_labeling' not in clustering_result:
            print(f"Warning: Clustering failed or returned unexpected result for k={k}. Skipping this k.")
            # Assign a very high WCSS/low BIC to penalize this option, or handle as per strategy
            # For simplicity, appending a value that will likely not be chosen as elbow
            if clustering_algorithm in ['GMM', 'SGMM']:
                 wcss_or_bic.append(np.inf) # For BIC, higher is worse (we negate later for elbow logic)
            else:
                 wcss_or_bic.append(np.inf) # For WCSS, higher is worse
            continue
            
        clustering_model = clustering_result['before_labeling']
        
        # Attempt to fit the model. Catch LinAlgError for GMM/SGMM if reg_covar wasn't enough.
        try:
            if not hasattr(clustering_model, 'fit_predict') and hasattr(clustering_model, 'fit'):
                 clustering_model.fit(X) # Some models might only have fit, then labels are separate
            # For models like GMM, fit() is usually called by fit_predict() if using pre_clustering
            # If pre_clustering already did fit_predict, labels are in clustering_result['model_labels']
            # The 'before_labeling' object should be the unfitted or partially fitted model for Elbow to work with.
            # Let's assume pre_clustering returns a model instance that Elbow can call fit() or access inertia_/bic() on.
            # If fit_predict was called in pre_clustering, inertia_/bic() might be available directly.
            # For consistency, let's assume we need to fit here if it wasn't fully done for Elbow purposes.
            # However, most pre_clustering functions for sklearn models do fit_predict.
            # The current pre_clustering_SGMM does fit_predict. So sgmm model is already fit.

        except np.linalg.LinAlgError as e:
            print(f"LinAlgError for k={k} with {clustering_algorithm}: {e}. Skipping this k.")
            if clustering_algorithm in ['GMM', 'SGMM']:
                 wcss_or_bic.append(np.inf)
            else:
                 wcss_or_bic.append(np.inf)
            continue
        except ValueError as e:
            # Catching other ValueErrors, e.g. if X has too few samples for k clusters
            print(f"ValueError for k={k} with {clustering_algorithm}: {e}. Skipping this k.")
            if clustering_algorithm in ['GMM', 'SGMM']:
                 wcss_or_bic.append(np.inf)
            else:
                 wcss_or_bic.append(np.inf)
            continue

        # Use appropriate score: inertia_ for KMeans-like, bic() for GMM/SGMM
        score = None
        if clustering_algorithm in ['GMM', 'SGMM']:
            if hasattr(clustering_model, 'bic'):
                score = clustering_model.bic(X)
            else:
                print(f"Warning: BIC not available for {clustering_algorithm} model at k={k}. Skipping score calculation.")
        else:
            if hasattr(clustering_model, 'inertia_'):
                score = clustering_model.inertia_
            else:
                # For models like DBSCAN, MShift, XMeans, GMeans, NeuralGas that don't have inertia_
                # We need a different metric for Elbow method. Silhouette score is an option but computationally expensive here.
                # Or, these algorithms might not be suitable for Elbow if they auto-determine k or don't expose inertia/BIC.
                # For now, if no inertia_, we can't use Elbow in its current form for these.
                # This part of Elbow_method is primarily designed for KMeans and GMM/SGMM.
                # Let's assume if it's not GMM/SGMM, it should have inertia_ or this Elbow logic is misapplied.
                print(f"Warning: Inertia not available for {clustering_algorithm} model at k={k}. Skipping score calculation.")
        
        if score is not None:
            wcss_or_bic.append(score)
        else:
            # If score could not be calculated, append a value that won't be chosen
            if clustering_algorithm in ['GMM', 'SGMM']:
                 wcss_or_bic.append(np.inf) 
            else:
                 wcss_or_bic.append(np.inf)

    if len(wcss_or_bic) < 2: # Need at least 2 points to calculate differences
        print("Warning: Not enough valid scores to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1 # Ensure k is at least 1
        if max_clusters == 0 : optimal_k =1 # avoid k=0
        return {
            'optimul_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }
    
    # Rate of change of slope; For GMM/SGMM, lower BIC is better → so reverse slope logic by finding max decrease (min -diff)
    # For WCSS (inertia), lower is better, so we want to find where the decrease starts to slow down (max second derivative)
    diff_scores = np.diff(wcss_or_bic)
    if len(diff_scores) < 1:
        print("Warning: Not enough score differences to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1
        if max_clusters == 0 : optimal_k =1
        return {
            'optimul_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }

    second_diff = np.diff(diff_scores) 
    if len(second_diff) == 0:
        # If only two k values were tested (e.g., k=2, k=3), second_diff will be empty.
        # Default to the k with the better score or a predefined k.
        # For GMM/SGMM (BIC, lower is better), choose k with min score.
        # For WCSS (inertia, lower is better), choose k with min score.
        optimal_k_index = np.argmin(wcss_or_bic)
        optimal_k = cluster_range[optimal_k_index] 
        print("Warning: Only two k values tested or not enough points for second derivative. Optimal k chosen by min score.")
    else:
        if clustering_algorithm in ['GMM', 'SGMM']:
            # For BIC, we look for the point where the BIC score starts to increase after decreasing, or decreases less steeply.
            # The elbow point is where the second derivative of BIC (w.r.t k) is maximized (most positive change in slope).
            # Since we want lower BIC, np.argmin(wcss_or_bic) would give the best k if no elbow.
            # Kneedle algorithm might be more robust here. For now, using second derivative approach.
            # We want the point before a significant increase, or where decrease significantly slows.
            # argmax(second_diff) means the largest increase in slope (from more negative to less negative, or from negative to positive slope of BIC diffs)
            optimal_k_index = np.argmax(second_diff) + 1 # +1 because second_diff is shorter by 1 than diff_scores
        else:
            # For WCSS (inertia), we look for the point where the rate of decrease slows down.
            # This corresponds to the maximum of the second derivative (largest positive value).
            optimal_k_index = np.argmax(second_diff) + 1 # +1 as above
        
        optimal_k = cluster_range[optimal_k_index] # Convert index back to k value

    return {
        'optimul_cluster_n': optimal_k,
        'best_parameter_dict': current_parameter_dict # Return the potentially updated dict
    }