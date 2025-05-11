# A machine to test clustering algorithm for labeling data and determine the performance of each clustering algorithm.

import argparse
import numpy as np
import time
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
from utils.cluster_adjust_mapping import cluster_mapping
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Dataset_Choose_Rule.save_csv import csv_compare_clustering, csv_compare_matrix_clustering
from Dataset_Choose_Rule.time_save import time_save_csv_VL


def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Interval_inverse")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette = args.eval_clustering_silhouette
    Association_mathod = args.association

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording


    # 1. Load data from csv
    start = time.time()

    file_path, file_number = file_path_line_nonnumber(file_type, file_number)
    # cut_type = str(input("Enter the data cut type: "))
    if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD', 'CICModbus23', 'CICModbus', 'MitM', 'Kitsune']:
        cut_type = 'random'
    else:
        cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    timing_info['1_load_data'] = time.time() - start


    # 2. Check data 'label'
    start = time.time()

    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if x.strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    timing_info['2_label_check'] = time.time() - start


    # 3. Feature-specific embedding and preprocessing
    start = time.time()

    data = time_scalar_transfer(data, file_type)

    # regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not
    regul = 'N'

    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)
    print("embedded_dataframe: ", embedded_dataframe)

    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
    print("mapped group: ", group_mapped_df)
    print("mapped_info: ", mapped_info_df)

    timing_info['3_embedding'] = time.time() - start


    # 4. Numpy(hstack) processing and PCA
    start = time.time()

    X = group_mapped_df
    columns_data = list(data.columns)
    columns_X = list(X.columns)
    diff_columns = list(set(columns_data) - set(columns_X))
    print("data-X col: ", diff_columns)


    if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus']:
        pca_want = 'N'
    else:
        pca_want = 'Y'

    # pca_want = str(input("\nDo you want to do PCA? (Y/n): "))
    if pca_want in ['Y', 'y']:
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            print("CANN is a classification, which means you need to use the full data.")
            X_reduced = X
        else:
            X_reduced = pca_func(X)
    else:
        X_reduced = X

    print(f"\n[DEBUG Data_Labeling.py] X_reduced (data used for clustering) shape: {X_reduced.shape}")
    if hasattr(X_reduced, 'columns'): # X_reduced is a DataFrame
        print(f"[DEBUG Data_Labeling.py] X_reduced columns: {list(X_reduced.columns)}")
    else: # X_reduced is a NumPy array
        print(f"[DEBUG Data_Labeling.py] X_reduced is a NumPy array (no direct column names). First 5 cols of first row: {X_reduced[0, :5] if X_reduced.shape[0] > 0 else 'empty'}")
    
    # Note: Information about X (group_mapped_df, before PCA) is also good to output
    if hasattr(X, 'columns'):
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) shape: {X.shape}")
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) columns: {list(X.columns)}")

    timing_info['4_pca'] = time.time() - start

    # Create original labels aligned with X_reduced
    # Assumption: Rows in 'data' DataFrame correspond to rows in 'X' (group_mapped_df),
    # and subsequently to rows in 'X_reduced' if X is a DataFrame OR if pca_func preserves row order from X (if X is NumPy).
    # The logs show data.shape[0], X.shape[0], and X_reduced.shape[0] are all the same (2504267).
    
    if 'label' not in data.columns:
        raise ValueError("'label' column is missing from 'data' DataFrame. Ensure labeling step (Step 2) is correct.")
    
    # Check if X (group_mapped_df) has an index that can be used to align with 'data'
    # If X was derived from 'data' and row order is preserved, direct use of data['label'] is fine.
    # If X involved row reordering or filtering inconsistent with data, a more robust alignment (e.g., using original indices) would be needed.
    # For now, we assume direct correspondence in row order and length.
    if len(data) != X_reduced.shape[0]:
        # This case should ideally not happen if data processing keeps row counts consistent
        raise ValueError(f"Row count mismatch: 'data' ({len(data)}) vs 'X_reduced' ({X_reduced.shape[0]}). Cannot reliably align labels.")
    
    original_labels_for_X_reduced = data['label'].to_numpy()
    print(f"[DEBUG Data_Labeling.py] 'original_labels_for_X_reduced' created - Shape: {original_labels_for_X_reduced.shape}, Unique values: {np.unique(original_labels_for_X_reduced, return_counts=True)}")


    # 5. Clustering and Mapping
    start = time.time()

    '''
    max_clusters_want = str(input("\nDo you need to enter the number of max_clusters? (Y/n): "))
    if max_clusters_want in ['Y', 'y']:
        max_clusters = int(input("\nEnter the desired number of max_clusters: "))
    else:
        print("\nThe number of max_clusters is set to the default value of 1000.")
        max_clusters = 1000
    '''
    max_clusters = 300
    
    # Hyperparameter_optimization = str(input("\nDo you need to do Hyperparameter_optimization? (Y/n): "))
    Hyperparameter_optimization = 'Y'
    if Hyperparameter_optimization in ['Y', 'y']:
        # print(f"[DEBUG Data_Labeling.py] 'data' to be passed to choose_clustering_algorithm - Shape: {data.shape}") # 이제 CNI에 직접 안 감
        # print(f"[DEBUG Data_Labeling.py] 'data' to be passed to choose_clustering_algorithm - Columns: {list(data.columns)}")
        
        # Pass X_reduced, original_labels_for_X_reduced, and other necessary params.
        # 'data' (original 117-col df) might still be needed by choose_clustering_algorithm for other purposes (e.g., Elbow method if it uses original data).
        clustering, GMM_type = choose_clustering_algorithm(data, X_reduced, original_labels_for_X_reduced, clustering_algorithm, max_clusters)

    elif Hyperparameter_optimization in ['N', 'n']:
        # --- BEGIN MODIFICATION 3 ---
        clustering, GMM_type = choose_clustering_algorithm_Non_optimization(data, X_reduced, original_labels_for_X_reduced, clustering_algorithm)
        # --- END MODIFICATION 3 ---
    else:
        raise Exception("You can only express your intent to proceed with hyperparameter tuning with Y/N.")
    data['cluster'] = clustering['Cluster_labeling']

    cluster_mapping(data)   # To verify the effect of 'clustering_nomal_identify' def

    timing_info['5_clustering'] = time.time() - start


    # 6. Evaluation Labeling
    start = time.time()

    if eval_clustering_silhouette == 'y':
        eval_clustering = evaluate_clustering(data['label'], data['cluster'], X_reduced)
        eval_clustering_adjust = evaluate_clustering(data['label'], data['adjusted_cluster'], X_reduced)
    elif eval_clustering_silhouette == 'n':
        eval_clustering = evaluate_clustering_wos(data['label'], data['cluster'])
        eval_clustering_adjust = evaluate_clustering_wos(data['label'], data['adjusted_cluster'])

    ''' # If input data has NaNs, we should consider the following fill-in-the-blank code
    data['adjusted_cluster'] = data['cluster'].map(cluster_mapping).fillna(-1).astype(int)

    # Filter out noise points (-1) for evaluation
    data_filtered = data[data['cluster'] != -1]

    # Evaluate clustering performance (Avg=macro)
    if not data_filtered.empty:
        ma_accuracy = accuracy_score(data_filtered['label'], data_filtered['cluster'])
        ma_precision = precision_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
        ma_recall = recall_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
        ma_f1 = f1_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
        ma_jaccard = jaccard_score(data_filtered['label'], data_filtered['cluster'], average='macro', zero_division=0)
        ma_silhouette = silhouette_score(X_reduced[data['cluster'] != -1], data_filtered['cluster'])
    else:
        ma_accuracy = ma_precision = ma_recall = ma_f1 = ma_jaccard = ma_silhouette = np.nan
    '''

    timing_info['6_evaluation'] = time.time() - start


    # 7. Save the results to csv file
    start = time.time()

    row_compare_df = csv_compare_clustering(file_type, clustering_algorithm, file_number, data, GMM_type)    # Not called, but saved.

    metrics_original = eval_clustering
    metrics_adjusted = eval_clustering_adjust
    metrics_df = csv_compare_matrix_clustering(file_type, file_number, clustering_algorithm, metrics_original, metrics_adjusted, GMM_type)
    print("\nThe result of the clustering score calculation: ")
    print(metrics_df)

    timing_info['7_save_result'] = time.time() - start


    # Full time history
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time

    # Save time information as a CSV
    time_save_csv_VL(file_type, file_number, clustering_algorithm, timing_info)


    return


if __name__ == '__main__':
    main()