# Identify nomal clusters and anomalous clusters with nomal data
# Input 'data' is initial data

import numpy as np
from utils.class_row import nomal_class_data


def clustering_nomal_identify(data, clusters, num_clusters):
    print(f"\n[DEBUG CNI] Received 'data' - Shape: {data.shape}, Columns: {list(data.columns)}")
    temp_kns_df = nomal_class_data(data)
    print(f"[DEBUG CNI] DataFrame for 'known_nomal_samples' (before .to_numpy()) - Shape: {temp_kns_df.shape}, Columns: {list(temp_kns_df.columns)}")
    known_nomal_samples = temp_kns_df.to_numpy()

    final_labels = np.zeros(len(data))   # Create an array (list) to store cluster labels
    threshold = 0.3 # Similar to Confidence

    for cluster_id in range(num_clusters):  # Repeat for the number of clusters
        cluster_mask = (clusters == cluster_id)  # Select data from that cluster

        # Empty cluster skip logic
        if not np.any(cluster_mask): # If no data points are currently assigned to the cluster_id
            # print(f"[INFO] Cluster {cluster_id} is empty. Skipping. Associated points in final_labels remain 0.")
            continue


        cluster_data = data[cluster_mask]  # Samples in a cluster Only
    
        # Even if cluster_data is verified to be non-empty above (np.any(cluster_mask) is True),
        # defend against edge cases where the cluster_data DataFrame itself may be empty (e.g. data indexing issues, etc.)
        if cluster_data.empty:
            # print(f"[INFO] Cluster {cluster_id} mask was not empty, but resulting cluster_data is empty. Skipping. Associated points in final_labels remain 0.")
            continue

        print(f"[DEBUG CNI] DataFrame for 'cluster_array' (cluster_id {cluster_id}, before .to_numpy()) - Shape: {cluster_data.shape}, Columns: {list(cluster_data.columns)}")
        cluster_array = cluster_data.to_numpy() # Quickly convert to Numpy Array

        # Calculate how much of that cluster data matches known_normal_samples
        # If any of the arrays being compared are empty, no comparison is performed and num_normal_in_cluster is set to 0.
        if cluster_array.size == 0 or known_nomal_samples.size == 0:
            num_normal_in_cluster = 0
        else:
            try:
                # Possible ValueError due to feature count mismatch
                # This part of the call should ensure that data (and known_normal_samples derived from it) and
                # ensure that the features in X (and the cluster_array derived from it) used for clustering match the features in data (and the known_normal_samples derived from it)
                num_normal_in_cluster = sum(
                    np.any(np.all(cluster_array[:, None] == known_nomal_samples, axis=2), axis=1)
                )
            except ValueError as e:
                print(f"[Error comparing cluster {cluster_id}] shape mismatch: {e}")
                print(f"  Cluster array shape: {cluster_array.shape}, Known normal samples shape: {known_nomal_samples.shape}")
                num_normal_in_cluster = 0
    
        # Calculate the normal data rate
        normal_ratio = num_normal_in_cluster / len(cluster_data) # if len(cluster_data) > 0 else 0 # This part of the condition is actually unnecessary
    
        # Returns Normal (0) or Anomalous (1) if the ratio is above the threshold
        cluster_label = 0 if normal_ratio >= threshold else 1
        final_labels[cluster_mask] = cluster_label  # Apply cluster-wide labels


    return final_labels