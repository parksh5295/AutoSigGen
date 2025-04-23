# Machines for detecting False positives

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 1. Apply a signature to create an alert (Optimized Vectorized Version)
def apply_signatures_to_dataset(df, signatures, base_time=datetime(2025, 4, 14, 12, 0, 0)):
    """
    Applies signatures to a DataFrame using vectorized operations for potentially faster performance.

    Args:
        df (pd.DataFrame): Input data, pre-processed (e.g., group mapped).
        signatures (list): List of signature dictionaries, each with 'id', 'name',
                           and 'rule_dict' containing the rule conditions.
        base_time (datetime): Base timestamp for alerts.

    Returns:
        pd.DataFrame: DataFrame containing generated alerts.
    """
    alerts = []
    # Preview the label column name in the source data
    label_col_name = None
    label_cols_present = []
    for col in ['label', 'class', 'Class']:
        if col in df.columns:
            label_cols_present.append(col)
            if label_col_name is None: # Use the first found label column
                label_col_name = col

    # Ensure input DataFrame index is unique if it's not already
    if not df.index.is_unique:
        print("Warning: DataFrame index is not unique. Resetting index.")
        df = df.reset_index(drop=True)


    # Initialize temporary columns to store results (indexes to calculate matched signature ID and time)
    # Use a temporary DataFrame to avoid modifying the original df if passed by reference elsewhere
    temp_df = pd.DataFrame(index=df.index)
    temp_df['_match_sig_id'] = pd.NA # Use pandas NA for better compatibility
    temp_df['_row_index'] = np.arange(len(df)) # For time calculation

    # Create signature_id and name mapping (pre-generate for faster lookup)
    sig_id_to_name = {s.get('id'): s.get('name', 'UNKNOWN_NAME') for s in signatures if s.get('id')}


    # Iterate through each signature condition and apply vectorized approach
    for sig_info in signatures:
        sig_id = sig_info.get('id', 'UNKNOWN_ID') # Use .get for safety

        # Check if 'rule_dict' key exists and is a dictionary
        if 'rule_dict' not in sig_info or not isinstance(sig_info['rule_dict'], dict):
             print(f"Warning: Skipping signature {sig_id} due to missing or invalid 'rule_dict'.")
             continue
        sig_condition_dict = sig_info['rule_dict']

        # Skip if the condition is empty
        if not sig_condition_dict:
            # print(f"Info: Skipping signature {sig_id} because its rule_dict is empty.")
            continue

        # Create a mask to find rows that satisfy all conditions (column=value)
        mask = pd.Series(True, index=df.index) # Start with all rows as True
        valid_signature = True # Signature validity flag
        try:
            for col, value in sig_condition_dict.items():
                if col in df.columns:
                    # Safely handle NaN values (NaN != value, NaN != NaN)
                    col_series = df[col] # Use original df for comparison
                    if pd.api.types.is_numeric_dtype(col_series) and pd.api.types.is_numeric_dtype(value):
                         # Compare numeric types safely
                         mask &= (col_series.astype(float) == float(value))
                    elif pd.isna(value):
                         mask &= col_series.isna()
                    else:
                         # General comparison using eq
                         mask &= col_series.eq(value)

                    # If mask becomes all False, no need to check further conditions
                    if not mask.any():
                        break
                else:
                    print(f"Warning: Column '{col}' needed by signature {sig_id} not found in DataFrame. Skipping this signature for matching.")
                    valid_signature = False
                    break

            if not valid_signature:
                 mask = pd.Series(False, index=df.index)

        except Exception as e:
             print(f"Warning: Error creating mask for signature {sig_id}: {e}")
             mask = pd.Series(False, index=df.index)


        # Record sig_id for rows that haven't matched any signature yet and satisfy current signature condition
        if mask.any():
            # Use temp_df for assigning match_sig_id
            match_indices = temp_df.index[mask & temp_df['_match_sig_id'].isna()]
            if not match_indices.empty:
                temp_df.loc[match_indices, '_match_sig_id'] = sig_id

    # Filter only matched alerts (Use temp_df to filter)
    alerts_df_raw = temp_df[temp_df['_match_sig_id'].notna()].copy()
    # Join back with original df to get necessary columns like labels
    # Ensure join works correctly even if index was reset
    alerts_df_raw = alerts_df_raw.join(df[[col for col in label_cols_present if col in df.columns]], lsuffix='_left')


    if alerts_df_raw.empty:
        print("Info: No alerts generated after applying all signatures.")
        return pd.DataFrame()

    print(f"Info: Generated {len(alerts_df_raw)} raw alerts.")

    # Create final alert DataFrame
    alerts_final = pd.DataFrame({
        'alert_index': alerts_df_raw.index,
        'timestamp': alerts_df_raw['_row_index'].apply(lambda i: base_time + timedelta(seconds=i * 2)),
        'src_ip': [f"192.168.1.{random.randint(1, 254)}" for _ in range(len(alerts_df_raw))],
        'dst_ip': [f"10.0.0.{random.randint(1, 254)}" for _ in range(len(alerts_df_raw))],
        'signature_id': alerts_df_raw['_match_sig_id'],
        'signature_name': alerts_df_raw['_match_sig_id'].map(sig_id_to_name)
    })

    # Copy label information from original data
    for col in label_cols_present:
         if col in alerts_df_raw.columns: # Check if column exists after join
            alerts_final[col] = alerts_df_raw[col].values
         else:
             print(f"Warning: Label column '{col}' not found in alerts_df_raw after join.")


    return alerts_final


# --- Start Replacement ---
# 2. Paper-based false positive determination (HAF Optimized, NRA slightly optimized)
def calculate_fp_scores(alerts_df: pd.DataFrame, attack_free_df: pd.DataFrame, 
                        t0_nra: int = 60, n0_nra: int = 20, 
                        lambda_haf: float = 100.0, 
                        lambda_ufp: float = 10.0, 
                        belief_threshold: float = 0.5,
                        combine='max'):
    """Calculates FP scores (NRA, HAF, UFP) with HAF optimized and NRA slightly improved."""
    if alerts_df.empty:
        print("Warning: calculate_fp_scores received an empty alerts_df. Returning empty DataFrame.")
        # Return DataFrame with expected columns for downstream compatibility
        # Make sure columns match those expected by evaluate_false_positives
        return pd.DataFrame(columns=['nra_score', 'haf_score', 'ufp_score', 'belief', 'is_false_positive', 'signature_id', 'timestamp', 'src_ip', 'dst_ip', 'signature_name']) # Added signature_name if needed later

    df = alerts_df.copy()
    # Ensure timestamp is datetime and handle potential errors
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting timestamp column to datetime: {e}. Attempting to continue, but results may be affected.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows where timestamp conversion failed
    initial_rows = len(df)
    df.dropna(subset=['timestamp'], inplace=True)
    if len(df) < initial_rows:
         print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid timestamps.")
    if df.empty:
        print("Warning: All rows dropped after handling invalid timestamps.")
        return pd.DataFrame(columns=['nra_score', 'haf_score', 'ufp_score', 'belief', 'is_false_positive', 'signature_id', 'timestamp', 'src_ip', 'dst_ip', 'signature_name'])

    # --- NRA Optimization Step 1: Sort DataFrame by timestamp BEFORE the loop ---
    print("Sorting alerts by timestamp for NRA calculation...")
    df.sort_values(by='timestamp', inplace=True)
    # Use reset_index() to get a simple integer index for progress tracking if needed
    # Keep original index in a column if it's meaningful and needed later
    df_original_index = df.index # Store original index if needed
    df.reset_index(drop=True, inplace=True) # Use simple 0-based index for iloc/iteration
    n = len(df)
    print("Sorting finished.")

    # --- 1. NRA Calculation (Using itertuples and pre-sorted data with searchsorted optimization) ---
    print("Calculating NRA scores (using itertuples with searchsorted)...")
    nra_scores = []
    # Need to make sure 'src_ip' and 'dst_ip' actually exist
    required_cols_nra = ['timestamp', 'src_ip', 'dst_ip']
    missing_cols = [col for col in required_cols_nra if col not in df.columns]
    if missing_cols:
        print(f"Error: Required columns for NRA {missing_cols} not found.")
        df['nra_score'] = 0.0 # Example: Set default NRA score
    else:
        # Create timestamp Series once for efficient filtering
        timestamps = df['timestamp']
        # Prepare src_ip and dst_ip Series (or arrays) for isin check
        src_ips = df['src_ip']
        dst_ips = df['dst_ip']

        # Loop using index for iloc access (can be faster than itertuples for large data)
        for i in range(n):
            # Access data using iloc for potentially better performance
            t_i = timestamps.iloc[i]
            src_ip_i = src_ips.iloc[i]
            dst_ip_i = dst_ips.iloc[i]

            ip_set = {src_ip_i, dst_ip_i}
            t_start = t_i - pd.Timedelta(seconds=t0_nra)
            t_end = t_i + pd.Timedelta(seconds=t0_nra)

            # Optimized window finding using searchsorted on sorted timestamps
            start_idx = timestamps.searchsorted(t_start, side='left')
            end_idx = timestamps.searchsorted(t_end, side='right')

            if start_idx >= end_idx: # Optimization: window is empty
                nra = 0
            else:
                # Slice the necessary IP columns for the window directly using iloc
                window_src_ips = src_ips.iloc[start_idx:end_idx]
                window_dst_ips = dst_ips.iloc[start_idx:end_idx]

                # Check if src_ip or dst_ip matches the current row's IPs using NumPy
                src_match_mask = np.logical_or(window_src_ips == src_ip_i, window_src_ips == dst_ip_i)
                dst_match_mask = np.logical_or(window_dst_ips == src_ip_i, window_dst_ips == dst_ip_i)

                # Combine masks: neighbor if either src or dst matches either IP
                combined_ip_mask = np.logical_or(src_match_mask, dst_match_mask)

                # Count neighbors directly from the boolean mask's sum
                nra = np.sum(combined_ip_mask)

        nra_scores.append(min(nra, n0_nra) / n0_nra)

            # Optional progress indicator using integer position 'i'
            if (i + 1) % 50000 == 0: # Print less frequently
                 print(f"  NRA progress: {i + 1}/{n}")

    df['nra_score'] = nra_scores

    # Restore original index if it was stored
    df.index = df_original_index

    print("NRA calculation finished.")

    # --- 2. HAF Calculation (Optimized using groupby and diff) ---
    print("Calculating HAF scores (vectorized)...")
    # Need signature_id and timestamp
    if 'signature_id' not in df.columns or 'timestamp' not in df.columns:
         print("Error: 'signature_id' or 'timestamp' columns missing for HAF calculation.")
         df['haf_score'] = 0.0 # Set default HAF score
    else:
        # Ensure sorting is done on the DataFrame with the correct index if needed later
        df_sorted_haf = df.sort_values(by=['signature_id', 'timestamp']).copy()
        df_sorted_haf['time_diff_prev'] = df_sorted_haf.groupby('signature_id')['timestamp'].diff().dt.total_seconds()
        df_sorted_haf['time_diff_next'] = df_sorted_haf.groupby('signature_id')['timestamp'].diff(-1).dt.total_seconds().abs()
        df_sorted_haf['mtd'] = df_sorted_haf[['time_diff_prev', 'time_diff_next']].abs().min(axis=1, skipna=True)
        df_sorted_haf['mtd'].fillna(np.inf, inplace=True)
        df_sorted_haf['fi'] = 1 / (1 + df_sorted_haf['mtd'])

        sig_stats = df_sorted_haf.groupby('signature_id')['timestamp'].agg(['min', 'max', 'count'])
        sig_stats['duration'] = (sig_stats['max'] - sig_stats['min']).dt.total_seconds()
        sig_stats['duration'] = sig_stats['duration'].clip(lower=0) # Ensure non-negative
        # Calculate avg_interval carefully for count=1 case
        sig_stats['avg_interval'] = np.where(sig_stats['count'] > 1, sig_stats['duration'] / (sig_stats['count'] - 1), np.inf)
        # Handle avg_interval=0 or inf for saf calculation
        sig_stats['saf'] = np.where(sig_stats['avg_interval'] > 1e-9, 1 / sig_stats['avg_interval'], np.inf)
        saf_map = sig_stats['saf']
        # Map saf back using the index of df_sorted_haf, then align with original df index
        df_sorted_haf['saf'] = df_sorted_haf['signature_id'].map(saf_map).fillna(np.inf) # Default saf to inf if not found


        df_sorted_haf['nf'] = df_sorted_haf['fi'] / df_sorted_haf['saf']
        df_sorted_haf['nf'].replace([np.inf, -np.inf, np.nan], 0, inplace=True) # Handle inf/nan from division

        df_sorted_haf['haf_score'] = (df_sorted_haf['nf'].clip(upper=lambda_haf) / lambda_haf)

        # Merge HAF scores back using the DataFrame's index (should align correctly now)
        # Ensure the index used for merging is the one from the original df
        df = df.merge(df_sorted_haf[['haf_score']], left_index=True, right_index=True, how='left')
        df['haf_score'].fillna(0, inplace=True)
    print("HAF calculation finished.")


    # --- 3. UFP Calculation (Optimized using map) ---
    print("Calculating UFP scores...")
    if attack_free_df is None or attack_free_df.empty or 'signature_id' not in attack_free_df.columns:
        print("Warning: attack_free_df is unsuitable for UFP. Scores set to 0.")
        df['ufp_score'] = 0.0
    elif 'signature_id' not in df.columns:
         print("Error: 'signature_id' column missing from main DataFrame for UFP.")
         df['ufp_score'] = 0.0
        else:
        af_counts = attack_free_df['signature_id'].value_counts()
        af_total = len(attack_free_df)
        af_freqs_map = (af_counts / af_total).to_dict() if af_total > 0 else {}
        test_counts_map = df['signature_id'].value_counts().to_dict()
        n_test = len(df)

        # Use map for potentially faster lookup
        test_freqs_series = df['signature_id'].map(test_counts_map).fillna(0) / n_test if n_test > 0 else pd.Series(0.0, index=df.index)
        af_freqs_series = df['signature_id'].map(af_freqs_map).fillna(1e-9) # Map known af freqs, default small

        # Calculate ratio avoiding division by zero
        ratio = np.where(af_freqs_series < 1e-9, lambda_ufp, test_freqs_series / af_freqs_series)
        df['ufp_score'] = np.minimum(ratio, lambda_ufp) / lambda_ufp
        df['ufp_score'].fillna(0, inplace=True) # Handle potential NaNs from calculation

    print("UFP calculation finished.")

    # --- 4. Determining combinations and false positives (No change) ---
    print("Combining scores...")
    valid_combine_methods = ['max', 'avg', 'min']
    if combine not in valid_combine_methods:
        print(f"Warning: Invalid combine method '{combine}'. Defaulting to 'max'.")
        combine = 'max' # Default to max

    score_cols = ['nra_score', 'haf_score', 'ufp_score']
    # Ensure score columns exist before attempting calculation
    missing_score_cols = [col for col in score_cols if col not in df.columns]
    if missing_score_cols:
        print(f"Error: Score columns {missing_score_cols} not found for combining.")
        df['belief'] = 0.0
        df['is_false_positive'] = True
    else:
        # Fill NaN in score columns before combining to avoid NaN results
        df[score_cols] = df[score_cols].fillna(0)
        if combine == 'max':
            df['belief'] = df[score_cols].max(axis=1)
        elif combine == 'avg':
            df['belief'] = df[score_cols].mean(axis=1)
        elif combine == 'min':
            df['belief'] = df[score_cols].min(axis=1)

    df['is_false_positive'] = df['belief'] < belief_threshold

    print("FP score calculation complete.")
    # Ensure all required columns are present before returning
    for col in score_cols + ['belief', 'is_false_positive']:
         if col not in df.columns:
              df[col] = 0 # Add missing columns with default value

    return df

# --- End Replacement ---


# 3. FP propensity summary by signature
def summarize_fp_by_signature(result_df: pd.DataFrame):
    summary = result_df.groupby(['signature_id', 'signature_name']).agg(
        total_alerts=('is_false_positive', 'count'),
        false_positives=('is_false_positive', 'sum')
    )
    summary['fp_rate'] = summary['false_positives'] / summary['total_alerts']
    return summary.sort_values('fp_rate', ascending=False)

def check_alert_frequency(alerts_df, time_window=3600, threshold_multiplier=3):
    """
    1. Check if adding a signature causes excessive alerts
    
    Args:
        alerts_df (DataFrame): Alert data
        time_window (int): Time window (seconds)
        threshold_multiplier (float): How many times the average to consider it excessive
    
    Returns:
        dict: Whether each signature has excessive alerts
    """
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    # Total monitoring period
    total_time = (alerts_df['timestamp'].max() - alerts_df['timestamp'].min()).total_seconds()
    total_windows = max(1, total_time / time_window)
    
    # Calculate the alert frequency per signature
    signature_counts = alerts_df['signature_id'].value_counts()
    average_alerts_per_window = signature_counts / total_windows
    
    # Calculate the number of alerts per hour
    alerts_per_hour = {}
    excessive_alerts = {}
    
    for sig_id in signature_counts.index:
        sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
        
        # Calculate the number of alerts per hour
        hourly_counts = sig_alerts.groupby(sig_alerts['timestamp'].dt.hour).size()
        max_hourly = hourly_counts.max()
        avg_hourly = hourly_counts.mean()
        
        alerts_per_hour[sig_id] = max_hourly
        # If the maximum number of alerts per hour is more than threshold_multiplier times the average
        excessive_alerts[sig_id] = max_hourly > (avg_hourly * threshold_multiplier)
    
    return excessive_alerts

def check_superset_signatures(new_signatures, known_fp_signatures):
    """
    2. Check if the newly created signature is a superset of existing FP signatures
    
    Args:
        new_signatures (list): List of new signatures
        known_fp_signatures (list): List of existing FP signatures
    
    Returns:
        dict: Whether each new signature is a superset
    """
    superset_check = {}
    
    for new_sig in new_signatures:
        # Check if the structure is as expected
        if 'signature_name' not in new_sig or not isinstance(new_sig['signature_name'], dict) or 'Signature_dict' not in new_sig['signature_name']:
            print(f"Warning: Skipping superset check for signature due to unexpected structure: {new_sig.get('id', 'N/A')}")
            continue 
        new_sig_dict = new_sig['signature_name']['Signature_dict']
        is_superset = False
        
        for fp_sig in known_fp_signatures:
             # Check FP signature structure too
             if 'signature_name' not in fp_sig or not isinstance(fp_sig['signature_name'], dict) or 'Signature_dict' not in fp_sig['signature_name']:
                 print(f"Warning: Skipping known FP signature in superset check due to unexpected structure.")
                 continue
            fp_sig_dict = fp_sig['signature_name']['Signature_dict']
            
            # Check if all conditions of fp_sig are included in new_sig
             if isinstance(new_sig_dict, dict) and isinstance(fp_sig_dict, dict): # Ensure they are dicts
            if all(k in new_sig_dict and new_sig_dict[k] == v for k, v in fp_sig_dict.items()):
                is_superset = True
                break
             else:
                  print(f"Warning: Invalid dictionary types for superset check. New: {type(new_sig_dict)}, FP: {type(fp_sig_dict)}")

        
        # Use signature ID if name is complex/missing
        sig_key = new_sig.get('id', new_sig.get('signature_name', str(new_sig))) 
        superset_check[sig_key] = is_superset
    
    return superset_check

def check_temporal_ip_patterns(alerts_df, time_window=300):
    """
    3. Check if similar source/destination IP alerts occur in a temporal manner when an attack occurs
    
    Args:
        alerts_df (DataFrame): Alert data
        time_window (int): Time window to check (seconds)
    
    Returns:
        dict: Pattern score for each signature
    """
    if 'timestamp' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: Required columns 'timestamp' or 'signature_id' not found for temporal pattern check.")
         return {}
         
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'], errors='coerce')
    alerts_df.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed
    if alerts_df.empty:
        return {}
        
    pattern_scores = {}
    
    # Pre-calculate required columns if they exist
    has_src_ip = 'src_ip' in alerts_df.columns
    has_dst_ip = 'dst_ip' in alerts_df.columns
    if not has_src_ip or not has_dst_ip:
        print("Warning: 'src_ip' or 'dst_ip' columns missing. IP pattern scores will be 0.")

    # Use groupby for potentially faster processing per signature
    for sig_id, sig_alerts_group in alerts_df.groupby('signature_id'):
        sig_alerts = sig_alerts_group.sort_values('timestamp').copy() # Sort within group
        
        if len(sig_alerts) < 2:
            pattern_scores[sig_id] = 0
            continue
        
        # Use rolling window or other vectorized approaches if possible?
        # Sticking to iterrows for now as rolling window logic is complex here
        ip_pattern_scores = []
        timestamps_np = sig_alerts['timestamp'].to_numpy() # For faster lookup maybe?

        for idx, alert in sig_alerts.iterrows():
            # Find window using searchsorted on the group's sorted timestamps
            t_i = alert['timestamp']
            t_start = t_i - pd.Timedelta(seconds=time_window)
            t_end = t_i + pd.Timedelta(seconds=time_window)

            # Find indices for the window within the current group
            start_idx = np.searchsorted(timestamps_np, t_start, side='left')
            end_idx = np.searchsorted(timestamps_np, t_end, side='right')
            
            # Slice the group DataFrame using iloc for the window
            window_alerts = sig_alerts.iloc[start_idx:end_idx]


            if window_alerts.empty or len(window_alerts) <= 1: # Need at least one other alert in window
                ip_pattern_scores.append(0) # Or just continue? Depends on desired score calculation
                continue

            # Calculate IP similarity only if IP columns exist
            same_src = 0
            same_dst = 0
            if has_src_ip:
            same_src = (window_alerts['src_ip'] == alert['src_ip']).sum()
            if has_dst_ip:
            same_dst = (window_alerts['dst_ip'] == alert['dst_ip']).sum()
            
            # Time proximity weight
            time_diffs = np.abs((window_alerts['timestamp'] - alert['timestamp']).dt.total_seconds())
            # Avoid division by zero if time_diff is exactly 0 (the alert itself)
            time_weights = 1 / (1 + time_diffs[time_diffs > 1e-9]) # Exclude self comparison potentially
            
            # Calculate IP pattern score
            window_size = len(window_alerts) # Already checked > 1
            ip_similarity = (same_src + same_dst) / (window_size * 2) if (has_src_ip or has_dst_ip) else 0
            # Handle case where time_weights might be empty if all diffs are zero
            time_density = time_weights.mean() if not time_weights.empty else 0
                pattern_score = (ip_similarity + time_density) / 2
                ip_pattern_scores.append(pattern_score)
        
        # Final pattern score for each signature
        pattern_scores[sig_id] = np.mean(ip_pattern_scores) if ip_pattern_scores else 0
    
    return pattern_scores

def is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts):
    """Verify that the current signature is a superset of one of the known FP signatures"""
    if not known_fp_sig_dicts or not isinstance(known_fp_sig_dicts, list):
        return False # False if there is no known FP
    if not current_sig_dict or not isinstance(current_sig_dict, dict):
        return False # False if there is no current signature

    for fp_sig_dict in known_fp_sig_dicts:
        if not isinstance(fp_sig_dict, dict): continue # Skip if FP signature format error

        # Check if all items (keys, values) in fp_sig_dict exist in current_sig_dict
        try:
            # Use items view for potentially faster check in Python 3
            is_superset = fp_sig_dict.items() <= current_sig_dict.items()
        except TypeError:
            # Handle cases where values might not be comparable (e.g., NaN)
            # Fallback to original check
        is_superset = all(
                 k in current_sig_dict and current_sig_dict[k] == v for k, v in fp_sig_dict.items()
            )

        # Original logic checked only subset, let's keep the superset check
        # is_superset = all(
        #    item in current_sig_dict.items() for item in fp_sig_dict.items()
        # )

        # Ensure it's a PROPER superset (current longer than fp)
        if is_superset and len(current_sig_dict) > len(fp_sig_dict):
             # print(f"Debug: {current_sig_dict} is superset of {fp_sig_dict}")
             return True
    return False

def evaluate_false_positives(
        alerts_df: pd.DataFrame,
        current_signatures_map: dict,
        known_fp_sig_dicts: list = None,
        attack_free_df: pd.DataFrame = None, # For UFP calculations
        # Add calculate_fp_scores parameters (set default values)
        t0_nra: int = 60,
        n0_nra: int = 20,
        lambda_haf: float = 100.0,
        lambda_ufp: float = 10.0,
        combine_method: str = 'max', # Changed from existing combine parameter (avoid Python reserved word conflict)
        # FP decision parameters
        belief_threshold: float = 0.5,
        superset_strictness: float = 0.9
    ):
    """
    Calculate FP scores and apply superset logic to determine final FP decision.
    """
    if alerts_df.empty:
         print("Warning: evaluate_false_positives received empty alerts_df. No FP analysis performed.")
         return pd.DataFrame(columns=['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive'])

    if attack_free_df is None:
         print("Warning: attack_free_df not provided for UFP calculation.")
         # Create an empty DataFrame with necessary columns to avoid errors in calculate_fp_scores
         attack_free_df = pd.DataFrame(columns=['signature_id'])


    # 1. Calculate basic FP scores (use received parameters)
    print("Calculating initial FP scores...")
    fp_scores_df = calculate_fp_scores(
        alerts_df,
        attack_free_df,
        t0_nra=t0_nra,
        n0_nra=n0_nra,
        lambda_haf=lambda_haf,
        lambda_ufp=lambda_ufp,
        combine=combine_method # Use modified parameter name
    )
    print("Initial FP score calculation finished.")

    # Check if necessary columns exist after calculate_fp_scores
    required_fp_cols = ['signature_id', 'belief', 'nra_score', 'haf_score', 'ufp_score']
    if not all(col in fp_scores_df.columns for col in required_fp_cols):
        print("Error: calculate_fp_scores did not return required columns. Cannot proceed.")
        return pd.DataFrame(columns=['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive'])


    # Initialize for saving results
    fp_results = fp_scores_df.copy()
    # Ensure required columns for the loop exist
    fp_results['is_superset'] = False
    fp_results['applied_threshold'] = belief_threshold
    fp_results['likely_false_positive'] = False

    # Add signature_name if it's missing (needed for summarize step)
    if 'signature_name' not in fp_results.columns and 'signature_id' in fp_results.columns:
         print("Adding signature_name based on current_signatures_map...")
         sig_id_to_name_map = {sig_id: sig_data.get('name', 'UNKNOWN') for sig_id, sig_data in current_signatures_map.items()}
         fp_results['signature_name'] = fp_results['signature_id'].map(sig_id_to_name_map)


    # If known FP list is not provided, initialize
    if known_fp_sig_dicts is None:
        known_fp_sig_dicts = []

    # 2. Check superset and determine final FP decision for each signature
    # Avoid iterrows if possible, maybe apply a function?
    # For now, keep iterrows but ensure it works correctly
    print("Applying superset logic and final FP decision...")
    num_rows_fp = len(fp_results)
    for index, row in fp_results.iterrows():
        sig_id = row['signature_id']
        belief_score = row['belief']

        # Get current signature dictionary
        current_sig_dict = current_signatures_map.get(sig_id)
        if not current_sig_dict:
            # print(f"Warning: Signature dictionary not found for {sig_id} in current_signatures_map.")
            continue # Skip if not in map

        # Check superset (using the potentially optimized is_superset_of_known_fps)
        is_super = is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts)
        fp_results.loc[index, 'is_superset'] = is_super

        # Apply threshold
        threshold = belief_threshold * superset_strictness if is_super else belief_threshold
        fp_results.loc[index, 'applied_threshold'] = threshold

        # Final FP decision
        fp_results.loc[index, 'likely_false_positive'] = belief_score < threshold

        # Optional progress
        # if (fp_results.index.get_loc(index) + 1) % 50000 == 0:
        #      print(f"  Superset/Final FP progress: {fp_results.index.get_loc(index) + 1}/{num_rows_fp}")

    print("Superset logic and final FP decision applied.")

    # Return final result DataFrame (before summarization)
    # Ensure all expected columns by summarize_fp_results are present
    final_cols = ['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive']
    missing_final_cols = [col for col in final_cols if col not in fp_results.columns]
    if missing_final_cols:
         print(f"Warning: Columns {missing_final_cols} missing before returning from evaluate_false_positives.")
         # Add missing columns with default values
         for col in missing_final_cols:
             if col == 'signature_name':
                 fp_results[col] = 'UNKNOWN'
             elif col == 'is_superset' or col == 'likely_false_positive':
                  fp_results[col] = False
             else:
                  fp_results[col] = 0.0


    return fp_results[final_cols]

def summarize_fp_results(detailed_fp_results: pd.DataFrame):
     """ Summarize FP decision results by group """
     if detailed_fp_results.empty:
          print("Info: detailed_fp_results is empty in summarize_fp_results.")
          return pd.DataFrame()

     # Ensure required columns exist
     required_summary_cols = ['signature_id', 'signature_name', 'likely_false_positive', 'belief', 'is_superset', 'applied_threshold']
     if not all(col in detailed_fp_results.columns for col in required_summary_cols):
          print(f"Error: Missing required columns in detailed_fp_results for summary. Need: {required_summary_cols}")
          return pd.DataFrame()


     summary = detailed_fp_results.groupby(['signature_id', 'signature_name']).agg(
         alerts_count=('likely_false_positive', 'size'), # Total alerts (group size)
         likely_fp_count=('likely_false_positive', 'sum'), # Number of alerts determined as FP
         avg_belief=('belief', 'mean'),
         is_superset=('is_superset', 'first'), # Superset status (same for each signature)
         applied_threshold=('applied_threshold', 'first') # Applied threshold
     ).reset_index()

     # Calculate FP rate safely avoiding division by zero
     summary['likely_fp_rate'] = np.where(summary['alerts_count'] > 0,
                                         summary['likely_fp_count'] / summary['alerts_count'],
                                         0)
     # Determine final FP status based on rate (e.g., > 50%)
     summary['final_likely_fp'] = summary['likely_fp_rate'] > 0.5

     # Ensure correct final columns are returned
     final_summary_cols = ['signature_id', 'signature_name', 'alerts_count', 'likely_fp_count', 'likely_fp_rate', 'avg_belief', 'is_superset', 'applied_threshold', 'final_likely_fp']

     # Add any missing columns from the final list if needed (shouldn't be necessary)
     for col in final_summary_cols:
         if col not in summary.columns:
              print(f"Warning: Column '{col}' missing in final summary. Adding default.")
              if col == 'signature_name':
                   summary[col] = 'UNKNOWN'
              elif col == 'is_superset' or col == 'final_likely_fp':
                   summary[col] = False
              else:
                   summary[col] = 0.0


     return summary[final_summary_cols]
