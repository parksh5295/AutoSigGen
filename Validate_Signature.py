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

# ===== Recall Calculation Helper Functions =====
def calculate_overall_recall(group_mapped_df, alerts_df, signature_map, relevant_signature_ids=None):
    '''
    Calculates the overall recall for a given set of signatures.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame returned by apply_signatures_to_dataset.
                                    Expected columns: 'alert_index', 'signature_id'.
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.
        relevant_signature_ids (set, optional): Set of signature IDs to consider.
                                                If None, all signatures in alerts_df are considered.

    Returns:
        float: Overall recall value (0.0 to 1.0).
    '''
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall calculation.")
        return 0.0
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall calculation.")
         return 0.0

    total_anomalous_alerts = group_mapped_df['label'].sum()
    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df.")
        return 0.0 # Avoid division by zero

    # Get indices of anomalous alerts in the original data
    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)

    # Filter alerts that correspond to anomalous original data
    anomalous_alerts_df = alerts_df[alerts_df['alert_index'].isin(anomalous_indices)].copy()

    # Filter by relevant signature IDs if provided
    if relevant_signature_ids is not None:
        print(f"Calculating recall based on {len(relevant_signature_ids)} signatures.")
        anomalous_alerts_df = anomalous_alerts_df[anomalous_alerts_df['signature_id'].isin(relevant_signature_ids)]
    else:
         print("Calculating recall based on all signatures present in alerts_df.")


    # Count unique anomalous alerts detected by the relevant signatures
    detected_anomalous_alerts = anomalous_alerts_df['alert_index'].nunique()

    recall = detected_anomalous_alerts / total_anomalous_alerts
    print(f"Total Anomalous Alerts: {total_anomalous_alerts}")
    print(f"Detected Anomalous Alerts (by relevant signatures): {detected_anomalous_alerts}")

    return recall


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
    parser.add_argument('--fp_belief_threshold', type=float, default=0.8)
    parser.add_argument('--fp_superset_strictness', type=float, default=0.9)
    parser.add_argument('--fp_t0_nra', type=int, default=60)
    parser.add_argument('--fp_n0_nra', type=int, default=20)
    parser.add_argument('--fp_lambda_haf', type=float, default=100.0)
    parser.add_argument('--fp_lambda_ufp', type=float, default=10.0)
    parser.add_argument('--fp_combine_method', type=str, default='max')
    parser.add_argument('--reset-known-fp', action='store_true',
                        help='Ignore existing known_high_fp_signatures.json and start fresh.')

    # Save the above in args
    args = parser.parse_args()

    # <<< START: Conditional FP Parameter Override for DARPA98 >>>
    if args.file_type in ["DARPA98", "DARPA"]:
        print("INFO: File type is 'DARPA98'. Applying specific stricter FP parameters.")
        # Override parameters only if they were not explicitly provided by the user
        # (We achieve this by checking if the current value is the standard default)
        if args.fp_belief_threshold == 0.5: # Standard default
             args.fp_belief_threshold = 0.95 # DARPA98 specific
        if args.fp_superset_strictness == 0.9:
             args.fp_superset_strictness = 0.6
        if args.fp_t0_nra == 60:
             args.fp_t0_nra = 180
        if args.fp_n0_nra == 20:
             args.fp_n0_nra = 100
        if args.fp_lambda_haf == 100.0:
             args.fp_lambda_haf = 25.0
        if args.fp_lambda_ufp == 10.0:
             args.fp_lambda_ufp = 2.5
    # <<< END: Conditional FP Parameter Override >>>

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
    # Use potentially overridden values from args
    fp_belief_threshold = args.fp_belief_threshold
    fp_superset_strictness = args.fp_superset_strictness
    fp_t0_nra = args.fp_t0_nra
    fp_n0_nra = args.fp_n0_nra
    fp_lambda_haf = args.fp_lambda_haf
    fp_lambda_ufp = args.fp_lambda_ufp
    fp_combine_method = args.fp_combine_method
    reset_known_fp = args.reset_known_fp

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
        signature_result = pd.DataFrame() # Initialize empty DataFrame if no signatures

    timing_info['4_basic_signature_evaluation'] = time.time() - start # Corrected timing key

    start = time.time() # Restart timer for FP/Overfitting

    # 2. False Positive check Preparation
    # Create a list of signatures formatted for the vectorized function
    formatted_signatures = [
        {
            'id': f'SIG_{idx}',
            'name': f'Signature_{idx}',
            # Store the actual rule dictionary instead of a lambda function
            'rule_dict': sig
        }
        for idx, sig in enumerate(signatures) # 'signatures' holds the original rule dicts
    ]

    # --- The following calls will eventually use the vectorized function ---
    # --- For now, the original apply_signatures_to_dataset might fail ---
    # --- because it expects 'condition' lambda, not 'rule_dict'.       ---
    # --- We will replace the function definition in the next step.     ---

    alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures)

    # Prepare data for FP analysis
    normal_data = group_mapped_df[group_mapped_df['label'] == 0].copy()
    attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures) # Alerts on normal data only

    # Traditional FP calculation (Optional: Keep for reference if needed)
    # fp_scores_traditional = calculate_fp_scores(alerts_df, attack_free_alerts) # Use original alerts_df for context if needed by func
    # fp_summary_traditional = summarize_fp_by_signature(fp_scores_traditional)
    # print("\n=== Traditional FP Summary (For Reference) ===")
    # print(fp_summary_traditional)

    # --- Load Known FP Signatures ---
    known_fp_sig_dicts = []
    if reset_known_fp:
        print("INFO: --reset-known-fp flag is set. Ignoring existing known FP file.")
    elif os.path.exists(KNOWN_FP_FILE):
        try:
            with open(KNOWN_FP_FILE, 'r') as f:
                known_fp_sig_dicts = json.load(f)
            print(f"Loaded {len(known_fp_sig_dicts)} known high-FP signatures from {KNOWN_FP_FILE}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {KNOWN_FP_FILE}. Starting with empty list.")
        except Exception as e:
            print(f"Warning: Error loading {KNOWN_FP_FILE}: {e}. Starting with empty list.")

    # --- Create current signature map (ID -> Dict) ---
    current_signatures_map = {
        f"SIG_{idx}": sig_dict
        for idx, sig_dict in enumerate(signatures)
    }
    initial_signature_ids = set(current_signatures_map.keys()) # All initial signature IDs set

    # --- Enhanced FP analysis ---
    print("\n=== False Positive analysis (Enhanced + Superset Logic) ===")
    fp_results_detailed = evaluate_false_positives(
        alerts_df.copy(), # Use all alerts for evaluation context
        current_signatures_map=current_signatures_map,
        known_fp_sig_dicts=known_fp_sig_dicts,
        attack_free_df=attack_free_alerts.copy(), # Use alerts strictly from normal data for FP identification
        belief_threshold=fp_belief_threshold,
        superset_strictness=fp_superset_strictness,
        t0_nra=fp_t0_nra,
        n0_nra=fp_n0_nra,
        lambda_haf=fp_lambda_haf,
        lambda_ufp=fp_lambda_ufp,
        combine_method=fp_combine_method,
        file_type=file_type
    )
    fp_summary_enhanced = summarize_fp_results(fp_results_detailed)

    # --- Add Signature Rule to Enhanced FP Summary ---
    # Ensure the column exists even if the dataframe is empty
    if 'signature_rule' not in fp_summary_enhanced.columns:
         fp_summary_enhanced['signature_rule'] = None

    if not fp_summary_enhanced.empty:
        fp_summary_enhanced['signature_rule'] = fp_summary_enhanced['signature_id'].map(current_signatures_map)
        # Convert dict to string for CSV compatibility if necessary, or handle during saving
        # fp_summary_enhanced['signature_rule'] = fp_summary_enhanced['signature_rule'].astype(str)

    print("Enhanced FP analysis results (summary):")
    if not fp_summary_enhanced.empty:
        # ===== Setting Pandas Output Options =====
        original_width = pd.get_option('display.width')
        original_max_colwidth = pd.get_option('display.max_colwidth')
        pd.set_option('display.width', 200) # Set a sufficiently wide width (adjustable)
        pd.set_option('display.max_colwidth', None) # Prevent column content truncation
        # ===============================

        print(fp_summary_enhanced.to_string(index=False)) # Cleaner output with index=False

        # ===== Restore original options (optional) =====
        pd.set_option('display.width', original_width)
        pd.set_option('display.max_colwidth', original_max_colwidth)
        # =====================================
    else:
        print("Enhanced FP summary results not found.")

    # --- Identify and report high FP signatures ---
    # Use 'final_likely_fp' returned from summarize_fp_results
    if not fp_summary_enhanced.empty and 'final_likely_fp' in fp_summary_enhanced.columns:
        newly_identified_fp = fp_summary_enhanced[fp_summary_enhanced['final_likely_fp']].copy()
        newly_identified_fp_ids = set(newly_identified_fp['signature_id'].tolist())
    else:
        print("Warning: Could not determine newly identified FP signatures. 'final_likely_fp' column missing or summary empty.")
        newly_identified_fp = pd.DataFrame() # Empty DataFrame
        newly_identified_fp_ids = set()

    newly_identified_fp_dicts = [
        current_signatures_map[sig_id] for sig_id in newly_identified_fp_ids if sig_id in current_signatures_map
    ]

    print(f"\nSignatures identified as high FP in this run: {len(newly_identified_fp_ids)}")
    if newly_identified_fp_ids:
        print("High FP signature IDs:", ", ".join(sorted(list(newly_identified_fp_ids))))
        # --- START: Print High FP Rules --- 
        print("--- Rules of High FP Signatures ---")
        for fp_id in sorted(list(newly_identified_fp_ids)):
            if fp_id in current_signatures_map:
                print(f"  ID: {fp_id}, Rule: {current_signatures_map[fp_id]}")
            else:
                print(f"  ID: {fp_id}, Rule: Not found in current map (unexpected error)")
        print("----------------------------------")
        # --- END: Print High FP Rules --- 
        # Print detailed information (optional)
        # print("Detailed information on newly identified High FP signatures:")
        # print(newly_identified_fp.to_string())

    # --- Update and save known FP list ---
    updated_known_fp_sig_dicts = known_fp_sig_dicts[:] # Create a copy
    existing_fp_strings = {json.dumps(d, sort_keys=True) for d in known_fp_sig_dicts}
    added_count = 0
    for new_fp_dict in newly_identified_fp_dicts:
         # Ensure the dict is serializable before dumping
         try:
             new_fp_string = json.dumps(new_fp_dict, sort_keys=True)
             if new_fp_string not in existing_fp_strings:
                 updated_known_fp_sig_dicts.append(new_fp_dict)
                 existing_fp_strings.add(new_fp_string)
                 added_count += 1
         except TypeError as e:
             print(f"Warning: Could not serialize signature for saving to known FP file: {new_fp_dict}. Error: {e}")


    if added_count > 0:
        print(f"{added_count} new high FP signatures identified to be saved.")
        ensure_directory_exists(KNOWN_FP_FILE) # Ensure directory exists before writing
        try:
            with open(KNOWN_FP_FILE, 'w') as f:
                json.dump(updated_known_fp_sig_dicts, f, indent=4)
            print(f"Updated known FP signature list saved: {KNOWN_FP_FILE}")
        except Exception as e:
            print(f"Error: Failed to save known FP signature list: {e}")
    else:
        print("No new high FP signatures to save.")

    # --- Overfitting check (Moved and Corrected) ---
    print("\n=== Overfitting score calculation ===")
    high_fp_signatures_count = len(newly_identified_fp_ids) # Use count from enhanced FP analysis
    total_signatures_count = len(signatures) # Use the initially loaded signatures

    overfit_results = evaluate_signature_overfitting(
        total_signatures_count=total_signatures_count,
        high_fp_signatures_count=high_fp_signatures_count
    )
    print_signature_overfit_report(overfit_results) # Use the calculated overfit_results

    # ===== Explicitly print Overfitting Score =====
    if 'overfitting_score' in overfit_results:
         print(f"Overall Overfitting Score: {overfit_results['overfitting_score']:.4f}")
    else:
         print("Could not determine overall overfitting score from results.")
    # ==========================================

    # --- Timing ---
    timing_info['5_fp_overfitting_check'] = time.time() - start # Combined timing step

    # --- Filter signatures (remove high FP) ---
    filtered_signatures_dicts = [
        sig_dict for idx, sig_dict in enumerate(signatures)
        if f"SIG_{idx}" not in newly_identified_fp_ids
    ]
    filtered_signature_ids = initial_signature_ids - newly_identified_fp_ids # Filtered signature IDs set
    print(f"\nOriginal signature count: {len(signatures)}")
    print(f"Signatures identified as high FP: {len(newly_identified_fp_ids)}")
    print(f"Filtered signature count: {len(filtered_signatures_dicts)}")
    # -----------------------------------

    # ===== Overall Recall Calculation and Output =====
    print("\n=== Overall Recall Calculation ===")
    # Assuming alerts_df contains 'alert_index' from apply_signatures_to_dataset
    if 'alert_index' in alerts_df.columns:
        print("\n--- Recall BEFORE FP Removal ---")
        recall_before_fp = calculate_overall_recall(group_mapped_df, alerts_df, current_signatures_map, initial_signature_ids)
        print(f"Recall (Before FP Removal): {recall_before_fp:.4f}")

        print("\n--- Recall AFTER FP Removal ---")
        if filtered_signature_ids:
            recall_after_fp = calculate_overall_recall(group_mapped_df, alerts_df, current_signatures_map, filtered_signature_ids)
            print(f"Recall (After FP Removal): {recall_after_fp:.4f}")
        else:
            print("No signatures left after filtering, Recall (After FP Removal): 0.0000")
            recall_after_fp = 0.0
    else:
        print("Warning: Cannot calculate overall recall because 'alert_index' is missing in alerts_df.")
        recall_before_fp = None
        recall_after_fp = None
    # =======================================

    # --- Evaluate performance with filtered signatures ---
    print("\n=== Evaluating Filtered Signatures ===")
    if filtered_signatures_dicts:
        # Re-use the signature_evaluate function
        # Need to re-format if signature_evaluate expects the original format
        # Assuming signature_evaluate can take list of dicts directly:
        # Call signature_evaluate and convert the resulting list to a DataFrame
        filtered_signature_result_list = signature_evaluate(group_mapped_df, filtered_signatures_dicts)
        filtered_signature_result = pd.DataFrame(filtered_signature_result_list) # <-- Convert list to DataFrame

        print("Filtered Signature Evaluation Results (first 5 rows):")
        # Now filtered_signature_result is a DataFrame, so .head() and .empty work
        print(filtered_signature_result.head().to_string() if not filtered_signature_result.empty else "No results")
    else:
        print("No signatures remaining after filtering.")
        # Define filtered_signature_result as empty DataFrame for consistent saving
        filtered_signature_result = pd.DataFrame()


    # --- Save all results to CSV ---
    print("\n--- Saving Validation Results ---")
    ensure_directory_exists(f"../Dataset/validation/{file_type}/") # Ensure save directory exists
    save_validation_results(
        file_type=file_type,
        file_number=file_number,
        association_rule=Association_mathod,
        basic_eval=signature_result, # Original evaluation results
        fp_results=fp_summary_enhanced, # Use enhanced FP summary (includes rules)
        overfit_results=overfit_results, # Use calculated overfitting results
        filtered_eval=filtered_signature_result, # <--- Delivering filtered assessment results
        recall_before=recall_before_fp,
        recall_after=recall_after_fp
    )

    # --- Save Timing Information ---
    timing_info['total_execution_time'] = time.time() - total_start_time
    ensure_directory_exists(f"../Dataset/Time_Record/validation/{file_type}/") # Ensure save directory exists
    time_save_csv_VS(file_type, file_number, Association_mathod, timing_info)


if __name__ == "__main__":
    main()

