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
                         mask &= (col_series.astype(float) == float(value))
                    elif pd.isna(value):
                         mask &= col_series.isna()
                    else:
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
    alerts_df_raw = alerts_df_raw.join(df[[col for col in label_cols_present if col in df.columns]])


    if alerts_df_raw.empty:
        print("Info: No alerts generated after applying all signatures.")
        return pd.DataFrame()

    print(f"Info: Generated {len(alerts_df_raw)} raw alerts.")

    # Create final alert DataFrame
    alerts_final = pd.DataFrame({
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

# 2. Paper-based false positive determination
def calculate_fp_scores(alerts_df: pd.DataFrame, attack_free_df: pd.DataFrame, 
                        t0_nra: int = 60, n0_nra: int = 20, 
                        lambda_haf: float = 100.0, 
                        lambda_ufp: float = 10.0, 
                        belief_threshold: float = 0.5,
                        combine='max'):
    df = alerts_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
    n = len(df)

    # 1. NRA
    nra_scores = []
    for i, row in df.iterrows():
        t_i = row['timestamp']
        ip_set = {row['src_ip'], row['dst_ip']}
        window = df[(df['timestamp'] >= t_i - pd.Timedelta(seconds=t0_nra)) &
                    (df['timestamp'] <= t_i + pd.Timedelta(seconds=t0_nra))]
        neighbors = window[
            (window['src_ip'].isin(ip_set)) | (window['dst_ip'].isin(ip_set))
        ]
        nra = len(neighbors)
        nra_scores.append(min(nra, n0_nra) / n0_nra)

    df['nra_score'] = nra_scores

    # 2. HAF
    signature_means = df.groupby('signature_id')['timestamp'].apply(
        lambda x: (x.max() - x.min()).total_seconds() / max(len(x) - 1, 1)
    ).to_dict()

    haf_scores = []
    for i, row in df.iterrows():
        sid = row['signature_id']
        t_i = row['timestamp']
        other = df[(df['signature_id'] == sid) & (df.index != i)]
        if not other.empty:
            time_diffs = np.abs((other['timestamp'] - t_i).dt.total_seconds())
            mtd = time_diffs.min()
            fi = 1 / (1 + mtd)
            saf = 1 / signature_means.get(sid, 1)
            nf = fi / saf
        else:
            nf = 0
        haf_scores.append(min(nf, lambda_haf) / lambda_haf)

    df['haf_score'] = haf_scores

    # 3. UFP
    af_freqs = attack_free_df['signature_id'].value_counts(normalize=True).to_dict()
    test_counts = df['signature_id'].value_counts().to_dict()

    ufp_scores = []
    for i, row in df.iterrows():
        sid = row['signature_id']
        f_af = af_freqs.get(sid, 1e-6)
        f_cur = test_counts.get(sid, 1e-6) / n
        ratio = f_cur / f_af
        ufp_scores.append(min(ratio, lambda_ufp) / lambda_ufp)

    df['ufp_score'] = ufp_scores

    # 4. Determining combinations and false positives
    if combine == 'max':
        df['belief'] = df[['nra_score', 'haf_score', 'ufp_score']].max(axis=1)
    elif combine == 'avg':
        df['belief'] = df[['nra_score', 'haf_score', 'ufp_score']].mean(axis=1)
    elif combine == 'min':
        df['belief'] = df[['nra_score', 'haf_score', 'ufp_score']].min(axis=1)
    else:
        raise ValueError("combine must be 'max', 'avg', or 'min'")

    df['is_false_positive'] = df['belief'] < belief_threshold
    return df

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
        new_sig_dict = new_sig['signature_name']['Signature_dict']
        is_superset = False
        
        for fp_sig in known_fp_signatures:
            fp_sig_dict = fp_sig['signature_name']['Signature_dict']
            
            # Check if all conditions of fp_sig are included in new_sig
            if all(k in new_sig_dict and new_sig_dict[k] == v for k, v in fp_sig_dict.items()):
                is_superset = True
                break
        
        superset_check[new_sig['signature_name']] = is_superset
    
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
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    pattern_scores = {}
    
    for sig_id in alerts_df['signature_id'].unique():
        sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id].copy()
        sig_alerts = sig_alerts.sort_values('timestamp')
        
        if len(sig_alerts) < 2:
            pattern_scores[sig_id] = 0
            continue
        
        # Analyze IP patterns within time window for each alert
        ip_pattern_scores = []
        for idx, alert in sig_alerts.iterrows():
            window_alerts = sig_alerts[
                (sig_alerts['timestamp'] >= alert['timestamp'] - pd.Timedelta(seconds=time_window)) &
                (sig_alerts['timestamp'] <= alert['timestamp'] + pd.Timedelta(seconds=time_window))
            ]
            
            # Calculate IP similarity
            same_src = (window_alerts['src_ip'] == alert['src_ip']).sum()
            same_dst = (window_alerts['dst_ip'] == alert['dst_ip']).sum()
            
            # Time proximity weight
            time_diffs = abs((window_alerts['timestamp'] - alert['timestamp']).dt.total_seconds())
            time_weights = 1 / (1 + time_diffs)
            
            # Calculate IP pattern score
            window_size = len(window_alerts)
            if window_size > 1:
                ip_similarity = (same_src + same_dst) / (window_size * 2)
                time_density = time_weights.mean()
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
        is_superset = all(
            item in current_sig_dict.items() for item in fp_sig_dict.items()
        )
        if is_superset and len(current_sig_dict) > len(fp_sig_dict): # Avoid strict subset (optional)
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
    if attack_free_df is None:
         print("Warning: attack_free_df not provided for UFP calculation.")
         attack_free_df = pd.DataFrame(columns=alerts_df.columns)

    # 1. Calculate basic FP scores (use received parameters)
    fp_scores_df = calculate_fp_scores(
        alerts_df,
        attack_free_df,
        t0_nra=t0_nra,
        n0_nra=n0_nra,
        lambda_haf=lambda_haf,
        lambda_ufp=lambda_ufp,
        combine=combine_method # Use modified parameter name
    )

    # Initialize for saving results
    fp_results = fp_scores_df.copy()
    fp_results['is_superset'] = False
    fp_results['applied_threshold'] = belief_threshold
    fp_results['likely_false_positive'] = False

    # If known FP list is not provided, initialize
    if known_fp_sig_dicts is None:
        known_fp_sig_dicts = []

    # 2. Check superset and determine final FP decision for each signature
    for index, row in fp_results.iterrows():
        sig_id = row['signature_id']
        belief_score = row['belief']

        # Get current signature dictionary
        current_sig_dict = current_signatures_map.get(sig_id)
        if not current_sig_dict:
            # print(f"Warning: Signature dictionary not found for {sig_id} in current_signatures_map.")
            continue # Skip if not in map

        # Check superset
        is_super = is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts)
        fp_results.loc[index, 'is_superset'] = is_super

        # Apply threshold
        threshold = belief_threshold * superset_strictness if is_super else belief_threshold
        fp_results.loc[index, 'applied_threshold'] = threshold

        # Final FP decision
        fp_results.loc[index, 'likely_false_positive'] = belief_score < threshold

    # Return final result DataFrame (before summarization)
    return fp_results[['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive']]

def summarize_fp_results(detailed_fp_results: pd.DataFrame):
     """ Summarize FP decision results by group """
     if detailed_fp_results.empty:
          return pd.DataFrame()

     summary = detailed_fp_results.groupby(['signature_id', 'signature_name']).agg(
         alerts_count=('likely_false_positive', 'size'), # Total alerts (group size)
         likely_fp_count=('likely_false_positive', 'sum'), # Number of alerts determined as FP
         avg_belief=('belief', 'mean'),
         is_superset=('is_superset', 'first'), # Superset status (same for each signature)
         applied_threshold=('applied_threshold', 'first') # Applied threshold
     ).reset_index()

     summary['likely_fp_rate'] = summary['likely_fp_count'] / summary['alerts_count']
     summary['final_likely_fp'] = summary['likely_fp_rate'] > 0.5 # Example: If more than half are FP, then final FP (adjustable)

     return summary[['signature_id', 'signature_name', 'alerts_count', 'likely_fp_count', 'likely_fp_rate', 'avg_belief', 'is_superset', 'applied_threshold', 'final_likely_fp']]
