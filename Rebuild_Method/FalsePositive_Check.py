# Machines for detecting False positives

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 1. Apply a signature to create an alert
def apply_signatures_to_dataset(df, signatures, base_time=datetime(2025, 4, 14, 12, 0, 0)):
    alerts = []

    for i, row in df.iterrows():
        for sig in signatures:
            try:
                if sig['condition'](row):
                    alerts.append({
                        'timestamp': base_time + timedelta(seconds=i * 2),
                        'src_ip': f"192.168.1.{random.randint(1, 254)}",
                        'dst_ip': f"10.0.0.{random.randint(1, 254)}",
                        'signature_id': sig['id'],
                        'signature_name': sig['name'],
                        'original_label': row.get('label', 'unknown')
                    })
                    break  # Apply only one signature
            except:
                continue  # To avoid conditional errors

    return pd.DataFrame(alerts)

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

def evaluate_false_positives(alerts_df, known_fp_signatures=None, 
                           time_window=3600, alert_threshold=3, 
                           pattern_threshold=0.7):
    """
    Main function to execute all FP checks
    
    Args:
        alerts_df (DataFrame): Alert data
        known_fp_signatures (list): List of existing FP signatures
        time_window (int): Time window (seconds)
        alert_threshold (float): Multiplier for alert threshold
        pattern_threshold (float): IP pattern score threshold
    
    Returns:
        DataFrame: FP evaluation results
    """
    # 1. Check excessive alerts
    excessive_alerts = check_alert_frequency(alerts_df, time_window, alert_threshold)
    
    # 2. Check superset (if known_fp_signatures is provided)
    superset_check = {}
    if known_fp_signatures:
        new_signatures = [{'signature_name': {'Signature_dict': sig}} 
                         for sig in alerts_df['signature_id'].unique()]
        superset_check = check_superset_signatures(new_signatures, known_fp_signatures)
    
    # 3. Check temporal IP patterns
    pattern_scores = check_temporal_ip_patterns(alerts_df)
    
    # 4. Combine results
    results = []
    for sig_id in alerts_df['signature_id'].unique():
        result = {
            'signature_id': sig_id,
            'excessive_alerts': excessive_alerts.get(sig_id, False),
            'is_superset': superset_check.get(sig_id, False) if known_fp_signatures else False,
            'ip_pattern_score': pattern_scores.get(sig_id, 0),
            'likely_false_positive': False
        }
        
        # FP determination logic
        result['likely_false_positive'] = (
            result['excessive_alerts'] or
            result['is_superset'] or
            result['ip_pattern_score'] < pattern_threshold
        )
        
        results.append(result)
    
    return pd.DataFrame(results)

def summarize_fp_results(fp_results):
    """
    Summary of FP evaluation results
    """
    summary = fp_results.groupby('signature_id').agg({
        'excessive_alerts': 'first',
        'is_superset': 'first',
        'ip_pattern_score': 'first',
        'likely_false_positive': 'first'
    })
    
    print("\n=== False Positive Analysis Summary ===")
    print(f"Total signatures evaluated: {len(summary)}")
    print(f"Signatures with excessive alerts: {summary['excessive_alerts'].sum()}")
    print(f"Signatures that are supersets of known FPs: {summary['is_superset'].sum()}")
    print(f"Signatures likely to be false positives: {summary['likely_false_positive'].sum()}")
    
    return summary
