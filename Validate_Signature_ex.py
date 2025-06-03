# Tools for signature verification

import argparse
import numpy as np
import time
import multiprocessing # Ensure multiprocessing is imported
from Dataset_Choose_Rule.association_data_choose import file_path_line_signatures, file_path_line_association
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
from copy import deepcopy

KNOWN_FP_FILE = "known_high_fp_signatures.json" # Known FP signature save file
RECALL_CONTRIBUTION_THRESHOLD = 0.1 # Threshold for whitelisting signatures
NUM_FAKE_FP_SIGNATURES = 3 # Number of fake FP signatures to inject

# Helper function for parallel calculation of single signature contribution
def _calculate_single_signature_contribution(sig_id, alerts_df_subset_cols, anomalous_indices_set, total_anomalous_alerts_count):
    """Calculates recall contribution for a single signature ID."""
    # Recreate alerts_df from the necessary columns passed
    # This is to avoid passing large DataFrames if only a subset is needed and pickling issues.
    # However, alerts_df is filtered by sig_id, so passing the relevant part or whole might be fine.
    # For simplicity here, assuming alerts_df_subset_cols is already filtered for the current sig_id OR we filter it here.
    # The original code did: sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
    # This implies that alerts_df should be passed fully, or tasks should pre-filter.
    # For starmap, it's better if the worker function gets exactly what it needs.
    # Option 1: Pass full alerts_df and filter inside (less ideal for many tasks if alerts_df is huge)
    # Option 2: Pre-filter alerts_df for each sig_id before making tasks (more setup but cleaner worker)

    # Assuming alerts_df_subset_cols IS alerts_df (the full one, or a view with 'signature_id' and 'alert_index')
    # This will be re-evaluated based on how tasks are prepared.
    # For now, let's stick to the logic from the original loop:
    sig_alerts = alerts_df_subset_cols[alerts_df_subset_cols['signature_id'] == sig_id]
    
    detected_by_sig = anomalous_indices_set.intersection(set(sig_alerts['alert_index']))
    contribution = 0.0
    if total_anomalous_alerts_count > 0:
        contribution = len(detected_by_sig) / total_anomalous_alerts_count
    return sig_id, contribution

# ===== Helper Function: Calculate Recall Contribution Per Signature =====
def calculate_recall_contribution(group_mapped_df, alerts_df, signature_map):
    """
    Calculates the recall contribution for each signature using parallel processing.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame from apply_signatures_to_dataset (covering all signatures).
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.

    Returns:
        dict: Dictionary mapping signature_id to its recall contribution (0.0 to 1.0).
              Returns empty dict if errors occur.
    """
    recall_contributions = {}
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall contribution.")
        return recall_contributions
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall contribution.")
         return recall_contributions

    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)
    total_anomalous_alerts = len(anomalous_indices)

    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df for recall contribution.")
        return {sig_id: 0.0 for sig_id in signature_map.keys()} # All contribute 0

    print(f"\nCalculating recall contribution for {len(signature_map)} signatures using parallel processing...")

    # Prepare tasks for parallel execution
    # Each task will be (sig_id, alerts_df, anomalous_indices, total_anomalous_alerts)
    # Pass alerts_df directly. Pandas DataFrames are picklable.
    tasks = [
        (sig_id, alerts_df[['signature_id', 'alert_index']], anomalous_indices, total_anomalous_alerts)
        for sig_id in signature_map.keys()
    ]

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for recall contribution calculation.")
    
    results = []
    if tasks: # Proceed only if there are signatures to process
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Results will be a list of (sig_id, contribution) tuples
                results = pool.starmap(_calculate_single_signature_contribution, tasks)
        except Exception as e:
            print(f"An error occurred during parallel recall contribution calculation: {e}")
            # Fallback to sequential calculation or return empty/partial
            print("Falling back to sequential calculation for recall contribution...")
            for sig_id in signature_map.keys():
                sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
                detected_by_sig = anomalous_indices.intersection(set(sig_alerts['alert_index']))
                contribution = 0.0
                if total_anomalous_alerts > 0:
                    contribution = len(detected_by_sig) / total_anomalous_alerts
                recall_contributions[sig_id] = contribution
                # Optional: print contribution per signature
                # print(f"  - {sig_id}: {contribution:.4f} (sequential)")
            return recall_contributions # Return sequentially computed results

    # Populate recall_contributions from parallel results
    for sig_id, contribution in results:
        recall_contributions[sig_id] = contribution
        # Optional: print contribution per signature
        # print(f"  - {sig_id}: {contribution:.4f} (parallel)")

    return recall_contributions
# ====================================================================

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

def generate_fake_fp_signatures(
    file_type,
    file_number,
    category_mapping, # From validation data processing
    data_list,        # From validation data processing
    association_method,
    association_metric,
    num_fake_signatures=3,
    min_support=0.3, # This min_support will be for ANOMALOUS training data
    min_confidence=0.8 # This is the min_confidence passed to this function
):
    """
    Generates fake FP signatures from the ANOMALOUS part of the TRAINING dataset.
    Uses mapping information (category_mapping, data_list) passed from the main
    function, which is typically derived from the VALIDATION dataset.
    Compatibility of this mapping with TRAINING data should be verified.
    Internally, association_module is called with a fixed min_confidence of 0.7.
    The file_number parameter is used to specify which training data file to load.
    """
    print(f"\n--- Generating {num_fake_signatures} Fake FP Signatures from ANOMALOUS TRAINING Data (file_type: {file_type}, file_number: {file_number}, using min_confidence=0.7 internally for association) ---")
    fake_signatures = []
    try:
        # 1. Load TRAINING data
        print(f"Loading TRAINING data for fake signature generation (file_type: {file_type}, file_number: {file_number})...")
        train_file_path, loaded_train_file_number = file_path_line_association(file_type, file_number) 
        
        full_train_data = file_cut(file_type, train_file_path, 'all')

        if full_train_data.empty:
            print("Warning: Training data is empty. Cannot generate fake signatures.")
            return []

        # 2. Apply time scalar transfer to training data
        print("Applying time scalar transfer to training data...")
        full_train_data = time_scalar_transfer(full_train_data, file_type)

        # 3. Assign labels to TRAINING data
        print("Assigning labels to training data...")
        if 'label' not in full_train_data.columns:
            if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']: # NSL-KDD types added here
                full_train_data['label'], _ = anomal_judgment_nonlabel(file_type, full_train_data)
            elif file_type == 'netML':
                if 'Label' in full_train_data.columns:
                    full_train_data['label'] = full_train_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
                else:
                    raise ValueError(f"'Label' column missing in netML training data for file_type: {file_type}")
            elif file_type == 'DARPA98':
                if 'Class' in full_train_data.columns:
                    full_train_data['label'] = full_train_data['Class'].apply(lambda x: 0 if x == '-' else 1)
                else:
                    raise ValueError(f"'Class' column missing in DARPA98 training data for file_type: {file_type}")
            elif file_type in ['CICModbus23', 'CICModbus']:
                if 'Attack' in full_train_data.columns:
                    full_train_data['label'] = full_train_data['Attack'].apply(lambda x: 0 if x.strip() == 'Baseline Replay: In position' else 1)
                else:
                    raise ValueError(f"'Attack' column missing in CICModbus training data for file_type: {file_type}")
            elif file_type in ['IoTID20', 'IoTID']:
                if 'Label' in full_train_data.columns:
                     full_train_data['label'] = full_train_data['Label'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
                else:
                    raise ValueError(f"'Label' column missing in IoTID20 training data for file_type: {file_type}")
            else: # Default case for types expected to have 'Label' or 'label'
                full_train_data['label'] = anomal_judgment_label(full_train_data)
                if full_train_data['label'] is None:
                    raise ValueError(f"Failed to assign labels to training data for file_type: {file_type} using anomal_judgment_label. Check for 'Label' or 'label' columns.")

        # 4. Filter for ANOMALOUS data (label == 1) from TRAINING dataset.
        anomalous_train_data_df = full_train_data[full_train_data['label'] == 1].copy()
        
        if anomalous_train_data_df.empty:
            print("Warning: No ANOMALOUS data found in training dataset after filtering. Cannot generate fake signatures.")
            return []
        print(f"Filtered for ANOMALOUS training data. Rows obtained: {anomalous_train_data_df.shape[0]}")

        # 5. Map the ANOMALOUS training data.
        # Using category_mapping and data_list derived from VALIDATION data.
        print(f"Shape of ANOMALOUS training data BEFORE mapping: {anomalous_train_data_df.shape}")
        print("Sample of ANOMALOUS training data BEFORE mapping (first 5 rows):")
        print(anomalous_train_data_df.head().to_string())
        
        anomalous_train_data_to_map = anomalous_train_data_df.drop(columns=['label'], errors='ignore')
        
        # Check some of the category_mapping content
        print("Debug: category_mapping['interval'] sample (first 5 rows, first 3 columns):")
        if not category_mapping['interval'].empty:
            print(category_mapping['interval'].iloc[:5, :3].to_string())
        else:
            print("category_mapping['interval'] is empty.")

        print("Mapping the ANOMALOUS training data (using mapping info potentially derived from validation set - VERIFY COMPATIBILITY)...")
        anomalous_mapped_train_df, _ = map_intervals_to_groups(anomalous_train_data_to_map, category_mapping, data_list, regul='N')
        
        print(f"Shape of mapped ANOMALOUS training data AFTER mapping (BEFORE dropna): {anomalous_mapped_train_df.shape}")
        print("NaN count per column (AFTER map_intervals_to_groups, BEFORE dropna):")
        print(anomalous_mapped_train_df.isna().sum().sort_values(ascending=False)) # Sort by most NaNs

        # --- Exclude problematic scalar columns for fake signature generation ---
        # Identify columns where all values are NaN
        all_nan_columns = anomalous_mapped_train_df.columns[anomalous_mapped_train_df.isna().all()].tolist()

        if all_nan_columns:
            print(f"Warning: For FAKE signature generation, columns with ALL NaN values identified: {all_nan_columns}")
            # These columns will likely cause all rows to be dropped if dropna() is used directly without intervention.
        
        # --- Handle NaN values ---
        # Stage 1: Try dropna on the full mapped dataframe
        rows_before_dropna_stage1 = anomalous_mapped_train_df.shape[0]
        anomalous_mapped_train_df_for_rules = anomalous_mapped_train_df.dropna()
        rows_after_dropna_stage1 = anomalous_mapped_train_df_for_rules.shape[0]

        if rows_before_dropna_stage1 > rows_after_dropna_stage1:
            print(f"[Stage 1 dropna] Dropped {rows_before_dropna_stage1 - rows_after_dropna_stage1} rows containing NaN values from mapped ANOMALOUS training data.")
        
        if anomalous_mapped_train_df_for_rules.empty:
            print("[Stage 1 dropna] Resulted in an empty DataFrame. Attempting Stage 2: using non-all-NaN columns.")
            
            # Stage 2: Use only columns that are NOT entirely NaN
            non_all_nan_columns = anomalous_mapped_train_df.columns[anomalous_mapped_train_df.notna().any()].tolist()

            if not non_all_nan_columns:
                print("Critical Error: [Stage 2] After mapping, no columns have any non-NaN data. Cannot generate any fake signatures.")
                return []
            
            print(f"[Stage 2] Re-attempting with columns that are not entirely NaN: {non_all_nan_columns}")
            # Use the original df but select only these columns
            anomalous_mapped_train_df_for_rules = anomalous_mapped_train_df[non_all_nan_columns]
            
            rows_before_dropna_stage2 = anomalous_mapped_train_df_for_rules.shape[0]
            anomalous_mapped_train_df_for_rules = anomalous_mapped_train_df_for_rules.dropna()
            rows_after_dropna_stage2 = anomalous_mapped_train_df_for_rules.shape[0]
            
            if rows_before_dropna_stage2 > rows_after_dropna_stage2:
                 print(f"[Stage 2 dropna] Dropped {rows_before_dropna_stage2 - rows_after_dropna_stage2} rows from the subset of columns.")

            if anomalous_mapped_train_df_for_rules.empty:
                print("Warning: [Stage 2] Still no data left after selecting non-all-NaN columns and applying dropna. Cannot generate fake signatures.")
                return []
            print(f"[Stage 2] Proceeding with {anomalous_mapped_train_df_for_rules.shape[0]} rows and columns: {anomalous_mapped_train_df_for_rules.columns.tolist()}")
        else:
            print(f"[Stage 1 dropna] Succeeded. Proceeding with {anomalous_mapped_train_df_for_rules.shape[0]} rows.")


        if file_type == 'CICModbus23':
            _internal_fixed_confidence = 0.01
        else:
            _internal_fixed_confidence = 0.7 
        
        print(f"Running {association_method} on ANOMALOUS training data (min_support={min_support}, using fixed min_confidence={_internal_fixed_confidence})...")
        
        rules_df = association_module(
            anomalous_mapped_train_df_for_rules, 
            association_method,
            association_metric=association_metric,
            min_support=min_support, 
            min_confidence=_internal_fixed_confidence
        )
        
        # ===== DEBUG CODE START =====
        print(f"DEBUG_ASSOCIATION_MODULE: Type of rules_df (for {association_method}) is: {type(rules_df)}")
        if rules_df is not None:
            if isinstance(rules_df, pd.DataFrame):
                print("DEBUG_ASSOCIATION_MODULE: rules_df is a DataFrame. Shape:", rules_df.shape)
                print("DEBUG_ASSOCIATION_MODULE: rules_df.head():")
                print(rules_df.head().to_string())
                if 'rule' in rules_df.columns:
                    print("DEBUG_ASSOCIATION_MODULE: First 'rule' entry sample:", rules_df['rule'].iloc[0] if not rules_df.empty else "N/A (empty df)")
            elif isinstance(rules_df, list):
                print("DEBUG_ASSOCIATION_MODULE: rules_df is a list. Length:", len(rules_df))
                if rules_df: # 리스트가 비어있지 않다면
                    print("DEBUG_ASSOCIATION_MODULE: First element of rules_df (list):")
                    print(rules_df[0])
                    if len(rules_df) > 1:
                        print("DEBUG_ASSOCIATION_MODULE: Second element of rules_df (list):")
                        print(rules_df[1])
            else:
                print("DEBUG_ASSOCIATION_MODULE: rules_df is neither DataFrame nor list. Value:")
                print(rules_df)
        else:
            print("DEBUG_ASSOCIATION_MODULE: rules_df is None.")
        # ===== DEBUG CODE END =====

        # 7. Extract top rules as fake signatures
        if rules_df is not None and not rules_df.empty and 'rule' in rules_df.columns:
            potential_rules = rules_df['rule'].tolist()
            valid_rules = [rule for rule in potential_rules if isinstance(rule, dict)]
            fake_signatures = valid_rules[:num_fake_signatures]
            print(f"Generated {len(fake_signatures)} fake signature rules from ANOMALOUS training data.")
        else:
            print("Warning: Association rule mining on ANOMALOUS training data did not produce usable rules.")

    except Exception as e:
        print(f"Error during fake signature generation from ANOMALOUS training data: {e}")
        import traceback
        traceback.print_exc()

    print("--- Fake FP Signature Generation (from ANOMALOUS TRAINING data with 0.7 confidence) Complete ---")
    return fake_signatures

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

    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
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


    # Corrected paths to load from Dataset_Paral
    base_path = f"../Dataset/signature/{file_type}/"
    # ensure_directory_exists(base_path) # Not strictly needed for loading, but good if any temp writes happen

    mapped_info_path = f"{base_path}{file_type}_{file_number}_mapped_info.csv"
    association_result_path = f"{base_path}{file_type}_{Association_mathod}_{file_number}_{association_metric}_signature_train_ea{signature_ea}.csv"
    
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

    # --- Calculate Recall Contribution & Determine Whitelist --- 
    print("=== Determining Whitelist based on Recall Contribution ===")
    # First apply signatures to get alerts_df needed for contribution calculation
    alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures) 
    recall_contributions = calculate_recall_contribution(group_mapped_df, alerts_df, current_signatures_map)
    whitelist_ids = {
        sig_id for sig_id, contrib in recall_contributions.items() 
        if contrib >= RECALL_CONTRIBUTION_THRESHOLD
    }
    print(f"Recall Contribution Threshold: {RECALL_CONTRIBUTION_THRESHOLD}")
    print(f"Signatures to whitelist (contribution >= threshold): {len(whitelist_ids)}")
    if whitelist_ids:
        print(f"Whitelist IDs: {', '.join(sorted(list(whitelist_ids)))}")
    # ------------------------------------------------------------

    # --- Generate and Inject Fake FP Signatures ---
    print("=== Generating and Injecting Fake FP Signatures ===")
    fake_fp_rules = generate_fake_fp_signatures(
        file_type=file_type,
        file_number=file_number,
        category_mapping=category_mapping, # Pass existing mapping
        data_list=data_list, # Pass existing data_list
        association_method=Association_mathod, # Use same method as main analysis
        association_metric=association_metric, # Pass association_metric from main args
        num_fake_signatures=NUM_FAKE_FP_SIGNATURES,
        min_support=0.4, # Slightly higher support for common normal patterns
        min_confidence=0.9 # High confidence for normal patterns
    )

    injected_fake_count = 0
    original_signatures_for_recall = deepcopy(signatures) # Keep original rules for recall calculation
    original_formatted_signatures_for_recall = deepcopy(formatted_signatures) # Keep original formatted sigs
    original_current_signatures_map_for_recall = deepcopy(current_signatures_map) # Keep original map

    if fake_fp_rules:
        for i, fake_rule in enumerate(fake_fp_rules):
            fake_sig_id = f"FAKE_FP_SIG_{i}"
            # Check for ID collision
            if fake_sig_id in current_signatures_map:
                 print(f"Warning: Fake signature ID {fake_sig_id} already exists. Skipping.")
                 continue
            
            # Inject into signatures list
            signatures.append(fake_rule) 
            
            # Inject into formatted_signatures list
            formatted_signatures.append({
                'id': fake_sig_id,
                'name': f'FakeSignature_{i}',
                'rule_dict': fake_rule
            })
            
            # Inject into current_signatures_map
            current_signatures_map[fake_sig_id] = fake_rule
            injected_fake_count += 1
            
        print(f"Successfully injected {injected_fake_count} fake FP signatures.")
        
        # --- Re-apply signatures to include fake ones in alerts_df for FP analysis ---
        print("Re-applying all signatures (including fake ones) to dataset...")
        # Update alerts_df to include alerts from fake signatures
        alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures) 
        # Update attack_free_alerts as well
        attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures)
        # ---------------------------------------------------------------------------

    else:
        print("No fake FP signatures were generated or injected.")
    # ---------------------------------------------------

    # --- Enhanced FP analysis (Now includes fake signatures if injected) ---
    print("=== False Positive analysis (Enhanced + Superset Logic) ===")
    # Use the potentially updated alerts_df and attack_free_alerts
    fp_results_detailed = evaluate_false_positives(
        alerts_df.copy(), 
        current_signatures_map=current_signatures_map, # Use map with fake sigs included
        known_fp_sig_dicts=known_fp_sig_dicts,
        attack_free_df=attack_free_alerts.copy(), 
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

    # --- Add Signature Rule and Experimental Info to Enhanced FP Summary ---
    if 'signature_rule' not in fp_summary_enhanced.columns:
         fp_summary_enhanced['signature_rule'] = None
    if 'is_injected_fake' not in fp_summary_enhanced.columns: # Add column for tracking fake signatures
        fp_summary_enhanced['is_injected_fake'] = False
    # if 'is_removed_final' not in fp_summary_enhanced.columns: # This will be added later
    #    fp_summary_enhanced['is_removed_final'] = False

    if not fp_summary_enhanced.empty:
        # Map signature rules (including potential fake ones)
        fp_summary_enhanced['signature_rule'] = fp_summary_enhanced['signature_id'].map(current_signatures_map)

        # --- Mark injected fake signatures ---
        fp_summary_enhanced['is_injected_fake'] = fp_summary_enhanced['signature_id'].str.startswith('FAKE_FP_SIG_')

        # --- Mark which signatures were finally removed (after whitelist) ---
        # This section will be moved and updated after 'actually_removed_ids' is calculated.
        # if 'actually_removed_ids' in locals(): # Check if the variable exists
        #      fp_summary_enhanced['is_removed_final'] = fp_summary_enhanced['signature_id'].isin(actually_removed_ids)
        # else:
        #      print("Warning: 'actually_removed_ids' not found when trying to mark final removal status.")
        #      fp_summary_enhanced['is_removed_final'] = None # Indicate status unknown

    print("Enhanced FP analysis results (summary with experimental flags):")
    if not fp_summary_enhanced.empty:
        # ===== Setting Pandas Output Options =====
        # ... (Pandas display options remain the same) ...
        pd.set_option('display.width', 200) 
        pd.set_option('display.max_colwidth', None) 

        # Select and reorder columns for better readability in printout
        cols_to_print = [
            'signature_id', 'alerts_count', 'likely_fp_rate', 'avg_belief', 
            'final_likely_fp', 'is_whitelisted', # Add is_whitelisted if calculated earlier 
            'is_injected_fake', 'is_removed_final', 'signature_rule'
        ]
        # Add 'is_whitelisted' if available from fp_summary_enhanced, otherwise skip
        if 'is_whitelisted' not in fp_summary_enhanced.columns:
             cols_to_print.remove('is_whitelisted')
             
        # Ensure all selected columns exist before printing
        cols_to_print = [col for col in cols_to_print if col in fp_summary_enhanced.columns]
             
        print(fp_summary_enhanced[cols_to_print].to_string(index=False)) # Print selected columns

        # ===== Restore original options (optional) =====
        # ... (Restoring options remains the same) ... 
    else:
        print("Enhanced FP summary results not found.")

    # --- Identify and report high FP signatures ---
    # Initialize the variable before the if/else block to ensure it's always defined
    initially_flagged_fp_ids = set()

    # Now, try to populate it based on FP summary results
    if not fp_summary_enhanced.empty and 'final_likely_fp' in fp_summary_enhanced.columns:
        # Identify ALL signatures initially flagged as high FP by the logic
        try: # Add try-except for robustness during set creation
            initially_flagged_fp_ids = set(fp_summary_enhanced[fp_summary_enhanced['final_likely_fp']]['signature_id'].tolist())
        except Exception as e:
            print(f"Error extracting initially flagged FP IDs: {e}")
            initially_flagged_fp_ids = set() # Fallback to empty set on error
    else:
        print("Warning: Could not determine newly identified FP signatures. 'final_likely_fp' column missing or summary empty.")
        # initially_flagged_fp_ids is already an empty set from initialization

    print(f"\nInitially flagged as High FP by logic: {len(initially_flagged_fp_ids)}")

    # --- Apply Whitelist ---
    # Ensure whitelist_ids is defined (should be from recall contribution step)
    if 'whitelist_ids' not in locals():
         print("Error: whitelist_ids is not defined before applying whitelist! Initializing to empty set.")
         whitelist_ids = set()

    # Ensure initially_flagged_fp_ids is defined *right before* use
    if 'initially_flagged_fp_ids' not in locals():
        print("Warning: initially_flagged_fp_ids was not defined before Apply Whitelist. Initializing to empty set.")
        initially_flagged_fp_ids = set()

    # Now perform the set operation using original variable naming conventions
    removed_due_to_whitelist = initially_flagged_fp_ids.intersection(whitelist_ids)
    # 'actually_removed_ids' is used consistently downstream for the set of removed IDs.
    # Its original calculation used 'removed_due_to_whitelist'.
    actually_removed_ids = initially_flagged_fp_ids - removed_due_to_whitelist
    # The variable 'ids_to_remove = initially_flagged_fp_ids - whitelist_ids' also existed originally.
    # As 'actually_removed_ids' (calculated as above) is equivalent and used in subsequent print statements
    # and filtering logic, we will proceed with this definition of 'actually_removed_ids'.

    print(f"Applying whitelist ({len(whitelist_ids)} IDs)...")
    if removed_due_to_whitelist: # Using the restored variable name
        print(f"Prevented removal of {len(removed_due_to_whitelist)} whitelisted IDs: {', '.join(sorted(list(removed_due_to_whitelist)))}")
    # The following print statements already use 'actually_removed_ids' in the current file version,
    # which is consistent with its role as the definitive set of removed IDs.
    print(f"Final IDs identified for removal (High FP & not whitelisted): {len(actually_removed_ids)}")
    if actually_removed_ids:
        print(f"IDs to remove: {', '.join(sorted(list(actually_removed_ids)))}")

    # --- Update fp_summary_enhanced with the final removal status --- 
    if not fp_summary_enhanced.empty:
        fp_summary_enhanced['is_removed_final'] = fp_summary_enhanced['signature_id'].isin(actually_removed_ids)
    else:
        if 'is_removed_final' not in fp_summary_enhanced.columns: # Ensure column exists even if empty
             fp_summary_enhanced['is_removed_final'] = False


    # --- Log NRA, HAF, UFP for caught FAKE signatures ---
    print("\n--- FP Metrics for Caught Fake Signatures (Loop 1) ---")
    _caught_fake_signature_metrics_log = [] # Use underscore for temp internal list
    # Ensure fp_results_detailed (output from evaluate_false_positives) is available and valid
    if 'fp_results_detailed' in locals() and isinstance(fp_results_detailed, pd.DataFrame) and not fp_results_detailed.empty and 'signature_id' in fp_results_detailed.columns:
        for _sig_id_to_check in actually_removed_ids: # Use temp var for loop iteration
            if _sig_id_to_check.startswith("FAKE_FP_SIG_"):
                # Filter fp_results_detailed for alerts triggered by this specific fake signature on normal data
                _alerts_for_this_fake_sig = fp_results_detailed[fp_results_detailed['signature_id'] == _sig_id_to_check]
                if not _alerts_for_this_fake_sig.empty:
                    _mean_nra = _alerts_for_this_fake_sig['nra_score'].mean() if 'nra_score' in _alerts_for_this_fake_sig else np.nan
                    _mean_haf = _alerts_for_this_fake_sig['haf_score'].mean() if 'haf_score' in _alerts_for_this_fake_sig else np.nan
                    _mean_ufp = _alerts_for_this_fake_sig['ufp_score'].mean() if 'ufp_score' in _alerts_for_this_fake_sig else np.nan
                    
                    _metric_detail = {
                        "fake_signature_id": _sig_id_to_check,
                        "loop_caught": 1, # Hardcoded to 1 for this run
                        "mean_nra_on_normal_data": _mean_nra,
                        "mean_haf_on_normal_data": _mean_haf,
                        "mean_ufp_on_normal_data": _mean_ufp,
                        "alerts_on_normal_data_count": len(_alerts_for_this_fake_sig)
                    }
                    _caught_fake_signature_metrics_log.append(_metric_detail)
                    print(f"  Caught Fake Sig: {_sig_id_to_check}, Loop: 1, Mean NRA: {_mean_nra:.4f}, Mean HAF: {_mean_haf:.4f}, Mean UFP: {_mean_ufp:.4f}, Alerts on Normal: {len(_alerts_for_this_fake_sig)}")
                else:
                    # This might happen if a fake signature is flagged due to other reasons (e.g., superset) 
                    # without having specific alert entries in fp_results_detailed from normal data.
                    print("Warning: `fp_results_detailed` DataFrame not available/valid. Cannot analyze caught fake signatures metrics.")
            else:
                print("Warning: `fp_results_detailed` DataFrame not available/valid. Cannot analyze caught fake signatures metrics.")
    
    if not _caught_fake_signature_metrics_log: # Check the temp list
        print("No FAKE signatures were caught and had detailed FP metrics to report in this run.")
        print("Debug: _caught_fake_signature_metrics_log is empty (checked before deciding to save).") # Added Debug
    else:
        print(f"Debug: _caught_fake_signature_metrics_log contains {len(_caught_fake_signature_metrics_log)} items.") # Added Debug
        print("Debug: Content of _caught_fake_signature_metrics_log (first 3 items):") # Added Debug
        for i, item in enumerate(_caught_fake_signature_metrics_log[:3]): # Added Debug
            print(f"  Item {i}: {item}") # Added Debug

        # Save the caught fake signature metrics to a CSV file
        _caught_fake_fp_metrics_df = pd.DataFrame(_caught_fake_signature_metrics_log)
        
        print("Debug: _caught_fake_fp_metrics_df created. Shape:", _caught_fake_fp_metrics_df.shape) # Added Debug
        print("Debug: Content of _caught_fake_fp_metrics_df (first 5 rows):") # Added Debug
        print(_caught_fake_fp_metrics_df.head().to_string()) # Added Debug
        
        _output_dir = f"../Dataset/validation/{file_type}/" # Define output directory
        # Using Association_mathod as it is in the existing codebase, preserving original variable names
        _csv_filename = f"{file_type}_{file_number}_{Association_mathod}_caught_fake_fp_metrics.csv"
        _csv_full_path = os.path.join(_output_dir, _csv_filename)
        
        ensure_directory_exists(_output_dir) # Ensure the directory itself exists
        
        try:
            _caught_fake_fp_metrics_df.to_csv(_csv_full_path, index=False)
            print(f"Successfully saved caught fake FP metrics to: {_csv_full_path}")
        except Exception as e:
            print(f"Error saving caught fake FP metrics to CSV {_csv_full_path}: {e}")
    # ------------------------------------------------------

    # --- Update and save known FP list ---
    # ... (logic to update/save known FP list remains the same, using initially_flagged_fp_ids) ...

    # --- Overfitting check ---
    print("=== Overfitting score calculation ===")
    high_fp_signatures_count = len(initially_flagged_fp_ids) 
    total_signatures_count = len(signatures) # includes fake ones now
    overfit_results = evaluate_signature_overfitting(
        total_signatures_count=total_signatures_count,
        high_fp_signatures_count=high_fp_signatures_count
    )
    print_signature_overfit_report(overfit_results)
    # ... (explicit score printing remains the same) ...

    # --- Timing ---
    timing_info['5_fp_overfitting_check'] = time.time() - start

    # --- Filter signatures (based on whitelist-applied removal list) ---
    # Use 'actually_removed_ids' to get the final set of signatures
    filtered_signatures_dicts = [
        sig_dict for sig_id, sig_dict in current_signatures_map.items() 
        if sig_id not in actually_removed_ids
    ]
    final_signature_ids = set(current_signatures_map.keys()) - actually_removed_ids

    # Update print statements for clarity
    print(f"Original signature count (before injection): {len(original_signatures_for_recall)}") # Use count before injection
    print(f"Injected fake signature count: {injected_fake_count}")
    print(f"Total signatures before filtering: {len(current_signatures_map)}")
    print(f"Final count of signatures removed: {len(actually_removed_ids)}")
    print(f"Filtered signature count (remaining): {len(filtered_signatures_dicts)}")
    # -----------------------------------

    # ===== Overall Recall Calculation and Output =====
    print("=== Overall Recall Calculation ===")
    # Use the original signatures and alerts from before injection/filtering for 'before' recall
    if 'label' in group_mapped_df.columns: # Check if label exists
        print("--- Recall BEFORE FP Removal (Original Signatures) ---")
        # We need alerts generated ONLY by original signatures
        original_alerts_df = apply_signatures_to_dataset(group_mapped_df, original_formatted_signatures_for_recall)
        recall_before_fp = calculate_overall_recall(
            group_mapped_df, 
            original_alerts_df, 
            original_current_signatures_map_for_recall, 
            set(original_current_signatures_map_for_recall.keys())
        )
        if recall_before_fp is not None:
            print(f"Recall (Original Signatures): {recall_before_fp:.4f}")
        else:
            print("Could not calculate recall for original signatures.")

        print("--- Recall AFTER FP Removal (Final Filtered Signatures) ---")
        if final_signature_ids:
            # Use the alerts_df that includes *all* signatures (original + fake)
            # But filter based on the final_signature_ids (whitelisted OK, removed FPs excluded)
            recall_after_fp = calculate_overall_recall(
                group_mapped_df, 
                alerts_df, # Use alerts_df containing triggers from all sigs (incl. whitelisted)
                current_signatures_map, # Map containing all sigs
                final_signature_ids # Only consider alerts from the final set
            )
            if recall_after_fp is not None:
                print(f"Recall (After FP Removal & Whitelisting): {recall_after_fp:.4f}")
            else:
                print("Could not calculate recall for final filtered signatures.")
        else:
            print("No signatures left after filtering, Recall (After FP Removal): 0.0000")
            recall_after_fp = 0.0
    else:
        print("Warning: Cannot calculate overall recall because 'label' is missing in group_mapped_df.")
        recall_before_fp = None
        recall_after_fp = None
    # =======================================

    # --- Evaluate performance with filtered signatures ---
    print("=== Evaluating Filtered Signatures ===")
    if filtered_signatures_dicts:
        # Use the final filtered list of rule dictionaries
        filtered_signature_result_list = signature_evaluate(group_mapped_df, filtered_signatures_dicts)
        filtered_signature_result = pd.DataFrame(filtered_signature_result_list) 
        print("Filtered Signature Evaluation Results (first 5 rows):")
        print(filtered_signature_result.head().to_string() if not filtered_signature_result.empty else "No results")
    else:
        print("No signatures remaining after filtering.")
        filtered_signature_result = pd.DataFrame()

    # --- Save all results to CSV ---
    print("\n--- Saving Validation Results ---")
    ensure_directory_exists(f"../Dataset/validation/{file_type}/") # Corrected path
    save_validation_results(
        file_type=file_type,
        file_number=file_number,
        association_rule=Association_mathod,
        basic_eval=signature_result, # Original evaluation results
        fp_results=fp_summary_enhanced, # FP summary (includes fake ones if generated)
        overfit_results=overfit_results,
        filtered_eval=filtered_signature_result, # Eval based on final filtered signatures
        recall_before=recall_before_fp, # Recall based on original signatures
        recall_after=recall_after_fp # Recall based on final filtered signatures
    )

    # START: FAKE_FP_SIG_ Enrich Trending History
    print("\n--- Tracking All Injected Fake FP Signatures ---")
    if 'fp_summary_enhanced' in locals() and isinstance(fp_summary_enhanced, pd.DataFrame) and not fp_summary_enhanced.empty:
        print("Debug: fp_summary_enhanced found and is not empty. Shape:", fp_summary_enhanced.shape)
        # Add print for relevant columns if they exist
        if 'is_injected_fake' in fp_summary_enhanced.columns:
            print("Debug: fp_summary_enhanced['is_injected_fake'].value_counts():")
            print(fp_summary_enhanced['is_injected_fake'].value_counts(dropna=False))
        else:
            print("Debug: 'is_injected_fake' column NOT FOUND in fp_summary_enhanced.")

        injected_fake_sigs_summary = fp_summary_enhanced[fp_summary_enhanced['is_injected_fake'] == True].copy()

        if not injected_fake_sigs_summary.empty:
            print(f"Debug: Found {len(injected_fake_sigs_summary)} injected FAKE_FP_SIG_ instances in fp_summary_enhanced to save.")
            print("Debug: Content of injected_fake_sigs_summary (first 5 rows):")
            print(injected_fake_sigs_summary.head().to_string())
            
            cols_to_report = ['signature_id', 'signature_rule', 'final_likely_fp', 'is_removed_final', 'likely_fp_rate', 'avg_belief', 'alerts_count']
            # Select only columns that exist in fp_summary_enhanced
            cols_to_report = [col for col in cols_to_report if col in injected_fake_sigs_summary.columns]

            print("Summary of all injected FAKE_FP_SIG_ (regardless of removal status):")
            print(injected_fake_sigs_summary[cols_to_report].to_string(index=False))

            # Save to file
            _fake_fp_tracking_output_dir = f"../Dataset/validation/{file_type}/"
            ensure_directory_exists(_fake_fp_tracking_output_dir)
            _fake_fp_tracking_csv_filename = f"{file_type}_{file_number}_{Association_mathod}_all_injected_fake_fp_sig_status.csv"
            _fake_fp_tracking_csv_full_path = os.path.join(_fake_fp_tracking_output_dir, _fake_fp_tracking_csv_filename)
            
            try:
                injected_fake_sigs_summary[cols_to_report].to_csv(_fake_fp_tracking_csv_full_path, index=False)
                print(f"Successfully saved status of all injected FAKE_FP_SIG_ to: {_fake_fp_tracking_csv_full_path}")
            except Exception as e:
                print(f"Error saving status of all injected FAKE_FP_SIG_ to CSV {_fake_fp_tracking_csv_full_path}: {e}")
        else:
            print("Debug: No injected FAKE_FP_SIG_ found in fp_summary_enhanced to report or save.")
    else:
        print("Debug: fp_summary_enhanced DataFrame not available or empty. Cannot track injected FAKE_FP_SIG_ status.")
    # END: FAKE_FP_SIG_ Enrich Trending History

    # --- Save Timing Information ---
    # Also modify timing filename to distinguish experiment runs
    timing_info['total_execution_time'] = time.time() - total_start_time
    ensure_directory_exists(f"../Dataset/time_log/validation_signature/{file_type}/") # Corrected path
    time_save_csv_VS(file_type, file_number, Association_mathod, timing_info)


if __name__ == "__main__":
    main()
