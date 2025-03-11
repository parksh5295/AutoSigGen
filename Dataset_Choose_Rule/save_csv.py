# for save Clustering, Association row to csv

import pandas as pd


def csv_compare(file_type, clusterint_method, file_number, data):
    row_compare_df = data[['cluster', 'adjusted_cluster', 'label']]
    row_compare_df.to_csv(f"../../Dataset/save_dataset/{file_type}/{file_type}_{clusterint_method}_{file_number}_clustering_Compare.csv", index=False)
    return row_compare_df

def csv_compare_matrix(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted):
    metrics_df = pd.DataFrame([metrics_original, metrics_adjusted], index=["Original", "Adjusted"])
    metrics_df.to_csv(f"../../Dataset/save_dataset/{file_type}/{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics.csv", index=True)
    return metrics_df