# for save Clustering, Association row to csv

import os
import pandas as pd


# Functions for creating folders if they don't exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def csv_compare_clustering(file_type, clusterint_method, file_number, data, GMM_type=None):
    row_compare_df = data[['cluster', 'adjusted_cluster', 'label']]
    
    save_path = f"../Dataset/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare.csv"
    row_compare_df.to_csv(file_path, index=False)
    
    return row_compare_df

def csv_compare_matrix_clustering(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type):
    metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
    
    save_path = f"../Dataset/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics.csv"
    metrics_df.to_csv(file_path, index=True)
    
    return metrics_df


def csv_association(file_type, file_number, association_rule, association_result, association_metric, signature_ea):
    df = pd.DataFrame([association_result])

    save_path = f"../Dataset/signature/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder

    file_path = f"{save_path}{file_type}_{association_rule}_{file_number}_{association_metric}_signature_train_ea{signature_ea}.csv"

    df.to_csv(file_path, index=False)
    