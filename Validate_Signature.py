# Tools for signature verification

import argparse
import numpy as np
import time
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from Dataset_Choose_Rule.dtype_optimize import load_csv_safely
from utils.class_row import anomal_class_data, without_labelmaking_out, nomal_class_data, without_label
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.remove_rare_columns import remove_rare_columns
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
from Evaluation.calculate_signature import calculate_signatures
from Modules.Difference_sets import dict_list_difference
from Dataset_Choose_Rule.save_csv import csv_association
from Dataset_Choose_Rule.time_save import time_save_csv_CS
import pandas as pd
from Modules.Signature_evaluation_module import signature_evaluate
from Rebuild_Method.FalsePositive_Check import evaluate_false_positives
from Rebuild_Method.Overfiting_Check import evaluate_signature_overfitting, print_signature_overfit_report
from Dataset_Choose_Rule.save_signature_validation import save_validation_results


def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--precision_underlimit', type=float, default=0.6)
    parser.add_argument('--signature_ea', type=int, default=15)
    parser.add_argument('--association_metric', type=str, default='confidence')

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
    precision_underlimit = args.precision_underlimit
    signature_ea = args.signature_ea
    association_metric = args.association_metric

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording


    # 1. Data loading
    start = time.time()

    file_path, file_number = file_path_line_signatures(file_type, file_number)
    # cut_type = str(input("Enter the data cut type: "))
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    timing_info['1_load_data'] = time.time() - start


    # 2. Handling judgments of Anomal or Nomal
    start = time.time()

    if file_type in ['MiraiBotnet']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    timing_info['2_anomal_judgment'] = time.time() - start

    data = time_scalar_transfer(data, file_type)


    start = time.time()


    mapped_info_path = f"../Dataset/signature/{file_type}/{file_type}_{file_number}_mapped_info.csv"
    association_result_path = f"../Dataset/signature/{file_type}/{file_type}_{association_rule}_{file_number}_{association_metric}_signature_train.csv"
    
    # Load data in an optimized way
    mapped_info_df = load_csv_safely('DARPA', mapped_info_path)
    association_result = pd.read_csv(association_result_path)

    # Extract mapping information from mapped_info_df
    category_mapping = {}
    for column in mapped_info_df.columns:
        mapping = {}
        for value in mapped_info_df[column].dropna().unique():
            if '=' in str(value):
                original, group = value.split('=')
                mapping[original.strip()] = int(group)
        if mapping:
            category_mapping[column] = mapping
    
    # Map data to groups
    group_mapped_df, _ = map_intervals_to_groups(data, category_mapping, list(category_mapping.keys()), regul='N')


    timing_info['3_group_mapping'] = time.time() - start

    
    start = time.time()


    # Extract signatures from association_result
    signatures = association_result['signature_name'].apply(lambda x: eval(x)['Signature_dict']).tolist()


    timing_info['4_signature_extraction'] = time.time() - start
    
    start = time.time()

    # 1. basic signature evaluation
    signature_result = signature_evaluate(mapped_info_df, signatures)
    print("\n=== Basic Signature Evaluation ===")
    print(signature_result)
    
    # 2. False Positive check
    fp_results = evaluate_false_positives(mapped_info_df, signatures)
    print("\n=== False Positive Analysis ===")
    print(fp_results)
    
    # 3. Overfitting check
    overfit_results = evaluate_signature_overfitting(mapped_info_df, signatures)
    print("\n=== Overfitting Analysis ===")
    print_signature_overfit_report(overfit_results)


    timing_info['5_signature_evaluation'] = time.time() - start


    # Save all results to CSV
    save_validation_results(
        file_type=file_type,
        file_number=file_number,
        association_rule=association_rule,
        basic_eval=signature_result,
        fp_results=fp_results,
        overfit_results=overfit_results
    )


    timing_info['total_execution_time'] = time.time() - total_start_time

    time_save_csv_CS(file_type, file_number, association_rule, timing_info)


if __name__ == "__main__":
    main()

