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
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset, calculate_fp_scores, summarize_fp_by_signature, evaluate_false_positives, summarize_fp_results
from Rebuild_Method.Overfiting_Check import evaluate_signature_overfitting, print_signature_overfit_report
from Dataset_Choose_Rule.save_signature_validation import save_validation_results
import ast  # Added for ast.literal_eval
import json
import os
import random # Add random import 
from datetime import datetime, timedelta # Add datetime import 

KNOWN_FP_FILE = "known_high_fp_signatures.json" # Known FP signature save file

def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)


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
    mapped_info_df = load_csv_safely(file_type, mapped_info_path)
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

    # Save label column separately before mapping
    label_series = data['label'] if 'label' in data.columns else None

    # Perform mapping
    group_mapped_df, _ = map_intervals_to_groups(data, category_mapping, data_list, regul='N')

    # Add the label from the source data to group_mapped_df
    group_mapped_df['label'] = data['label']

    print("\nVerifying label addition:")
    print(f"Original data shape: {data.shape}")
    print(f"Mapped data shape: {group_mapped_df.shape}")
    print(f"Label column exists in mapped data: {'label' in group_mapped_df.columns}")

    # Signature evaluation
    timing_info['3_group_mapping'] = time.time() - start

    
    start = time.time()


    # Extract signatures from association_result
    signatures = []
    verified_sigs = ast.literal_eval(association_result['Verified_Signatures'].iloc[0])
    if isinstance(verified_sigs, list):
        try:
            # Evaluate string to Python object
            sig_list = verified_sigs  # This will be a list
            
            # Extract Signature_dict from each signature
            for sig in sig_list:
                if isinstance(sig, dict) and 'signature_name' in sig:
                    sig_info = sig['signature_name']
                    if isinstance(sig_info, dict) and 'Signature_dict' in sig_info:
                        signatures.append(sig_info['Signature_dict'])
            
            print(f"Found {len(signatures)} valid signatures")
            
        except Exception as e:
            print(f"Error parsing signatures: {e}")
    else:
        print(f"Unexpected type for Verified_Signatures: {type(verified_sigs)}")

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
    
    # Add new FP check logic
    fp_results = evaluate_false_positives(
        alerts_df,
        time_window=3600,
        alert_threshold=3,
        pattern_threshold=0.7
    )
    
    # Keep the original FP check logic
    normal_data = group_mapped_df[group_mapped_df['label'] == 0].copy()
    attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures)

    fp_scores = calculate_fp_scores(alerts_df, attack_free_alerts)
    fp_summary = summarize_fp_by_signature(fp_scores)

    print("\n=== False Positive Analysis ===")
    print("Traditional FP Summary by Signature:")
    print(fp_summary)
    print("\nEnhanced FP Analysis Results:")
    print(fp_results[['signature_id', 'excessive_alerts', 'ip_pattern_score', 'likely_false_positive']])
    
    # 3. Overfitting check
    high_fp_sig_ids = set(high_fp_sigs['signature_id'].tolist()) # Calculated in the previous step
    high_fp_signatures_count = len(high_fp_sig_ids)
    total_signatures_count = len(initial_signatures_dicts) # Total number of initial signatures

    print("\n=== Overfitting score calculation ===") # Step name changed
    # Modified function call (pass count)
    overfit_results = evaluate_signature_overfitting(
        total_signatures_count=total_signatures_count,
        high_fp_signatures_count=high_fp_signatures_count
    )
    print_signature_overfit_report(overfit_results) # modified report function call


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

    # Load a known FP signature
    known_fp_sig_dicts = []
    if os.path.exists(KNOWN_FP_FILE):
        try:
            with open(KNOWN_FP_FILE, 'r') as f:
                known_fp_sig_dicts = json.load(f)
            print(f"Loaded {len(known_fp_sig_dicts)} known high-FP signatures from {KNOWN_FP_FILE}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {KNOWN_FP_FILE}. Starting with empty list.")
        except Exception as e:
            print(f"Warning: Error loading {KNOWN_FP_FILE}: {e}. Starting with empty list.")

    # Create current signature map (ID -> Dict)
    current_signatures_map = {
        f"SIG_{idx}": sig_dict
        for idx, sig_dict in enumerate(signatures)
    }

    # --- FP analysis (modified function call) ---
    print("\n=== False Positive analysis (Enhanced + Superset Logic) ===")
    fp_results_detailed = evaluate_false_positives(
        alerts_df.copy(),
        current_signatures_map=current_signatures_map, # Pass current signature map
        known_fp_sig_dicts=known_fp_sig_dicts,      # Pass known FP list
        belief_threshold=0.5,                       # Default threshold
        superset_strictness=0.9,                    # Superset strictness
        attack_free_df=attack_free_alerts          # Pass attack_free_df for UFP calculation
    )
    fp_summary_enhanced = summarize_fp_results(fp_results_detailed) # Call summary function

    print("Enhanced FP analysis results (summary):")
    if not fp_summary_enhanced.empty:
        print(fp_summary_enhanced.to_string())
    else:
        print("Enhanced FP summary results not found.")
    # ---------------------------------

    # --- Identify and report high FP signatures ---
    # Use 'final_likely_fp' returned from summarize_fp_results
    newly_identified_fp = fp_summary_enhanced[fp_summary_enhanced['final_likely_fp']]
    newly_identified_fp_ids = set(newly_identified_fp['signature_id'].tolist())
    newly_identified_fp_dicts = [
        current_signatures_map[sig_id] for sig_id in newly_identified_fp_ids if sig_id in current_signatures_map
    ]

    print(f"\nSignatures identified as high FP in this run: {len(newly_identified_fp_ids)}")
    if newly_identified_fp_ids:
        print("High FP signature IDs:", ", ".join(sorted(list(newly_identified_fp_ids))))
        # Print detailed information (optional)
        # print("Detailed information:")
        # print(newly_identified_fp)

    # --- Update and save known FP list ---
    # Merge existing list with newly identified list (remove duplicates)
    # Dictionaries cannot be directly converted to sets, so we need to use a different method
    updated_known_fp_sig_dicts = known_fp_sig_dicts[:] # Create a copy
    existing_fp_strings = {json.dumps(d, sort_keys=True) for d in known_fp_sig_dicts}
    added_count = 0
    for new_fp_dict in newly_identified_fp_dicts:
         new_fp_string = json.dumps(new_fp_dict, sort_keys=True)
         if new_fp_string not in existing_fp_strings:
             updated_known_fp_sig_dicts.append(new_fp_dict)
             existing_fp_strings.add(new_fp_string)
             added_count += 1

    if added_count > 0:
        print(f"{added_count} new high FP signatures saved.")
        try:
            with open(KNOWN_FP_FILE, 'w') as f:
                json.dump(updated_known_fp_sig_dicts, f, indent=4)
            print(f"Updated known FP signature list saved: {KNOWN_FP_FILE}")
        except Exception as e:
            print(f"Error: Failed to save known FP signature list: {e}")
    else:
        print("No new high FP signatures to save.")
    # ---------------------------------------

    # --- Filter signatures (remove high FP) ---
    # Use newly_identified_fp_ids
    filtered_signatures_dicts = [
        sig_dict for idx, sig_dict in enumerate(signatures)
        if f"SIG_{idx}" not in newly_identified_fp_ids
    ]
    print(f"Signatures removed (high FP): {len(filtered_signatures_dicts)}")
    # -----------------------------------

    # ... (After removal, performance evaluation and comparison - keep previous answer code) ...
    # Calculate initial_overfit_results
    # Calculate filtered_overfit_results
    # Print performance comparison

    # ... (Save final results - keep previous answer code) ...
    # Consider passing fp_results=fp_summary_enhanced to save_validation_results

    # ... (Save timing information - keep previous answer code) ...


if __name__ == "__main__":
    main()

