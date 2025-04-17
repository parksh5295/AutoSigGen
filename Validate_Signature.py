# Tools for signature verification

import argparse
import numpy as np
import time
from Dataset_Choose_Rule.association_data_choose import file_path_line_signatures
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
from Dataset_Choose_Rule.time_save import time_save_csv_VS
import pandas as pd
from Modules.Signature_evaluation_module import signature_evaluate
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset, calculate_fp_scores, summarize_fp_by_signature
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
    association_result_path = f"../Dataset/signature/{file_type}/{file_type}_{Association_mathod}_{file_number}_{association_metric}_signature_train_ea{signature_ea}.csv"
    
    # Load data in an optimized way
    mapped_info_df = load_csv_safely('DARPA', mapped_info_path)
    print("Loading association result from:", association_result_path)
    association_result = pd.read_csv(association_result_path)

    # Extract mapping information from mapped_info_df
    category_mapping = {
        'interval': {},
        'categorical': pd.DataFrame(),
        'binary': pd.DataFrame()
    }

    # Process interval mapping
    for column in mapped_info_df.columns:
        column_mappings = []
        for value in mapped_info_df[column].dropna():  # Process only non-NaN values
            if isinstance(value, str) and '=' in value:  # If mapping information exists
                column_mappings.append(value)
        
        if column_mappings:  # If mapping exists, add it
            category_mapping['interval'][column] = pd.Series(column_mappings)

    # Convert to DataFrame
    category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])

    # Create data_list - list of DataFrames with empty columns
    data_list = [pd.DataFrame(), pd.DataFrame()]  # Add empty DataFrames at the beginning and end

    # Perform mapping
    group_mapped_df, _ = map_intervals_to_groups(data, category_mapping, data_list, regul='N')


    timing_info['3_group_mapping'] = time.time() - start

    
    start = time.time()


    # Extract signatures from association_result
    signatures = []
    for sig_info in association_result['signature_name']:
        if isinstance(sig_info, str):
            try:
                # Evaluate string to Python object
                sig_dict = eval(sig_info)
                if isinstance(sig_dict, dict) and 'Signature_dict' in sig_dict:
                    signatures.append(sig_dict['Signature_dict'])
            except:
                print(f"Error parsing signature: {sig_info}")

    print("\nExtracted signatures:", signatures)

    # 1. basic signature evaluation
    if signatures:
        signature_result = signature_evaluate(group_mapped_df, signatures)
    else:
        print("Error: No valid signatures found")
    
    # 2. False Positive check
    formatted_signatures = [
        {
            'id': f'SIG_{idx}',
            'name': f'Signature_{idx}',
            'condition': lambda row, sig=sig: all(
                row[k] == v for k, v in sig.items()
            )
        }
        for idx, sig in enumerate(signatures)
    ]

    alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures)
    normal_data = group_mapped_df[group_mapped_df['label'] == 0].copy()
    attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures)

    fp_scores = calculate_fp_scores(alerts_df, attack_free_alerts)
    fp_summary = summarize_fp_by_signature(fp_scores)

    print("\n=== False Positive Analysis ===")
    print("FP Summary by Signature:")
    print(fp_summary)
    
    # 3. Overfitting check
    overfit_results = evaluate_signature_overfitting(group_mapped_df, signatures)
    print("\n=== Overfitting Analysis ===")
    print_signature_overfit_report(overfit_results)


    timing_info['5_signature_evaluation'] = time.time() - start


    # Save all results to CSV
    save_validation_results(
        file_type=file_type,
        file_number=file_number,
        association_rule=Association_mathod,
        basic_eval=signature_result,
        fp_results=fp_summary,
        overfit_results=overfit_results
    )


    timing_info['total_execution_time'] = time.time() - total_start_time

    time_save_csv_VS(file_type, file_number, Association_mathod, timing_info)


if __name__ == "__main__":
    main()

