# Adjust Cluster Labels to Match Ground Truth


def cluster_mapping(data):
    cluster_mapping = {0: 1, 1: 0}
    data['adjusted_cluster'] = data['cluster'].map(cluster_mapping)
    return