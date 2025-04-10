# Identify nomal clusters and anomalous clusters with nomal data
# Input 'data' is initial data

import numpy as np
from utils.class_row import nomal_class_data


def clustering_nomal_identify(data, clusters, num_clusters):
    known_nomal_samples = nomal_class_data(data).to_numpy()

    final_labels = np.zeros(len(data))   # Create an array (list) to store cluster labels
    threshold = 0.1 # Similar to Confidence

    for cluster_id in range(num_clusters):  # Repeat for the number of clusters
        cluster_mask = (clusters == cluster_id)  # Select data from that cluster
        cluster_data = data[cluster_mask]  # Samples in a cluster Only
    
        cluster_array = cluster_data.to_numpy() # Quickly convert to Numpy Array

        # Calculate how much of that cluster data matches known_normal_samples
        try:
            num_normal_in_cluster = sum(
                np.any(np.all(cluster_array[:, None] == known_nomal_samples, axis=2), axis=1)
            )
        except ValueError as e:
            print(f"[Error comparing cluster {cluster_id}] shape mismatch: {e}")
            num_normal_in_cluster = 0
    
        # Calculate the normal data rate
        normal_ratio = num_normal_in_cluster / len(cluster_data) if len(cluster_data) > 0 else 0
    
        # Returns Normal (0) or Anomalous (1) if the ratio is above the threshold
        cluster_label = 0 if normal_ratio >= threshold else 1
        final_labels[cluster_mask] = cluster_label  # Apply cluster-wide labels


    return final_labels