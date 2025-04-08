# Manage all pre_clustering_* functions in one place

import importlib


# Function that takes a clustering algorithm name and returns the corresponding pre_clustering_* function
def get_clustering_function(clustering_algorithm):
    # Standardized name mapping table
    CLUSTERING_NAME_MAP = {
        'kmeans': 'Kmeans',
        'kmedians': 'Kmedians',
        'gmm': 'GMM',
        'sgmm': 'SGMM',
        'gmeans': 'Gmeans',
        'xmeans': 'Xmeans',
        'dbscan': 'DBSCAN',
        'mshift': 'MShift',
        'fcm': 'FCM',
        'ck': 'CK',
        'neuralgas': 'NeuralGas',
        'cannwknn': 'CANNwKNN',
        'cann': 'CANNwKNN'
    }

    # Takes a clustering algorithm name, dynamically imports the corresponding pre_clustering_* function, and returns it.
    normalized_name = clustering_algorithm.lower()  #NEEDTOFIX
    
    if normalized_name not in CLUSTERING_NAME_MAP:
        raise ValueError(f"Unsupported clustering algorithm: {clustering_algorithm}")

    standard_name = CLUSTERING_NAME_MAP[normalized_name]

    module_name = f"Clustering_Method.clustering_{standard_name}"   # Ex. clustering_Kmeans
    function_name = f"pre_clustering_{standard_name}"   # Ex. pre_clustering_Kmeans 

    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)  # Get functions from that module
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found. Check if the clustering algorithm is implemented.")
    except AttributeError:
        raise ImportError(f"Function '{function_name}' not found in module '{module_name}'.")
