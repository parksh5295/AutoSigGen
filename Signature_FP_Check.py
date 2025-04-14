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
