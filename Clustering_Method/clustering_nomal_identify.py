# Identify nomal clusters and anomalous clusters with nomal data
# Input 'data' is initial data

import numpy as np
from utils.class_row import nomal_class_data


def clustering_nomal_identify(data, clusters, num_clusters):
    known_nomal_samples = nomal_class_data(data)

    cluster_labels = np.zeros(num_clusters)   # Create an array (list) to store cluster labels
    threshold = 0.5 # Similar to Confidence

    for cluster_id in range(num_clusters):  # Repeat for the number of clusters
        cluster_mask = (clusters == cluster_id)  # Select data from that cluster
        cluster_data = data[cluster_mask]  # Samples in a cluster Only
    
        # Calculate how much of that cluster data matches known_normal_samples
        num_normal_in_cluster = sum(np.any(np.all(cluster_data[:, None] == known_nomal_samples, axis=2), axis=1))
    
        # Calculate the normal data rate
        normal_ratio = num_normal_in_cluster / len(cluster_data) if len(cluster_data) > 0 else 0
    
        # Returns Normal (0) or Anomalous (1) if the ratio is above the threshold
        cluster_label = 0 if normal_ratio >= threshold else 1
        cluster_labels[cluster_mask] = cluster_label  # Apply cluster-wide labels


    return cluster_labels[cluster_mask]