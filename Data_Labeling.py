# A machine to test clustering algorithm for labeling data and determine the performance of each clustering algorithm.

import argparse
import numpy as np
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judment_label
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
from utils.cluster_adjust_mapping import cluster_mapping
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Dataset_Choose_Rule.save_csv import csv_compare_clustering, csv_compare_matrix_clustering


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


    # 1. Load data from csv
    file_path = file_path_line_nonnumber(file_type, file_number)
    cut_type = str(input("Enter the data cut type: "))
    data = file_cut(file_path, cut_type)


    # 2. Check data 'label'
    if file_type == 'MiraiBotnet':
        data['label'] = anomal_judgment_nonlabel(file_type, data)
    else:
        data['label'] = anomal_judment_label(data)


    # 3. Feature-specific embedding and preprocessing
    regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not

    embedded_dataframe, feature_list, category_mapping = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)

    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, regul)
    print("mapped group: ", group_mapped_df)
    print("mapped_info: ", mapped_info_df)


    # 4. Numpy(hstack) processing and PCA
    X = group_mapped_df

    pca_want = str(input("\nDo you want to do PCA? (Y/n): "))
    if pca_want in ['Y', 'y']:
        X_reduced = pca_func(X)
    else:
        X_reduced = X


    # 5. Clustering and Mapping
    max_clusters_want = str(input("\nDo you need to enter the number of max_clusters? (Y/n): "))
    if max_clusters_want in ['Y', 'y']:
        max_clusters = int(input("\nEnter the desired number of max_clusters: "))
    else:
        print("\nThe number of max_clusters is set to the default value of 1000.")
        max_clusters = 1000
    
    Hyperparameter_optimization = str(input("\nDo you need to do Hyperparameter_optimization? (Y/n): "))
    if Hyperparameter_optimization in ['Y', 'y']:
        clustering = choose_clustering_algorithm(data, X_reduced, clustering_algorithm, max_clusters)
    elif Hyperparameter_optimization in ['N', 'n']:
        clustering = choose_clustering_algorithm_Non_optimization(data, X_reduced, clustering_algorithm)
    else:
        raise Exception("You can only express your intent to proceed with hyperparameter tuning with Y/N.")
    data['cluster'] = clustering['Cluster_labeling']

    cluster_mapping(data)   # To verify the effect of 'clustering_nomal_identify' def


    # 6. Evaluation Labeling
    if eval_clustering_silhouette == 'y':
        eval_clustering = evaluate_clustering(data['label'], data['cluster'], X_reduced)
        eval_clustering_adjust = evaluate_clustering(data['label'], data['adjusted_cluster'], X_reduced)
    elif eval_clustering_silhouette == 'n':
        eval_clustering = evaluate_clustering_wos(data['label'], data['cluster'], X_reduced)
        eval_clustering_adjust = evaluate_clustering_wos(data['label'], data['adjusted_cluster'], X_reduced)

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


    # 7. Save the results to csv file
    row_compare_df = csv_compare_clustering(file_type, clustering_algorithm, file_number, data)    # Not called, but saved.

    metrics_original = eval_clustering
    metrics_adjusted = eval_clustering_adjust
    metrics_df = csv_compare_matrix_clustering(file_type, file_number, clustering_algorithm, metrics_original, metrics_adjusted)
    print("\nThe result of the clustering score calculation: ")
    print(metrics_df)


    return


if __name__ == '__main__':
    main()