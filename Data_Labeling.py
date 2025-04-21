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
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    timing_info['1_load_data'] = time.time() - start


    # 2. Check data 'label'
    start = time.time()

    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    timing_info['2_label_check'] = time.time() - start


    # 3. Feature-specific embedding and preprocessing
    start = time.time()

    data = time_scalar_transfer(data, file_type)

    regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not

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


    if file_type in [DARPA98, DARPA]:
        pca_want = 'Y'
    else:
        pca_want = 'N'

    # pca_want = str(input("\nDo you want to do PCA? (Y/n): "))
    if pca_want in ['Y', 'y']:
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            print("CANN is a classification, which means you need to use the full data.")
            X_reduced = X
        else:
            X_reduced = pca_func(X)
    else:
        X_reduced = X

    timing_info['4_pca'] = time.time() - start


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
    
    Hyperparameter_optimization = str(input("\nDo you need to do Hyperparameter_optimization? (Y/n): "))
    if Hyperparameter_optimization in ['Y', 'y']:
        clustering, GMM_type = choose_clustering_algorithm(data, X_reduced, clustering_algorithm, max_clusters)
    elif Hyperparameter_optimization in ['N', 'n']:
        clustering, GMM_type = choose_clustering_algorithm_Non_optimization(data, X_reduced, clustering_algorithm)
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