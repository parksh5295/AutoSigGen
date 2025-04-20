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
                    alert = {
                        'timestamp': base_time + timedelta(seconds=i * 2),
                        'src_ip': f"192.168.1.{random.randint(1, 254)}",
                        'dst_ip': f"10.0.0.{random.randint(1, 254)}",
                        'signature_id': sig['id'],
                        'signature_name': sig['name'],
                    }
                    # Copy all label columns from the original data
                    for label_col in ['label', 'class', 'Class']:
                        if label_col in df.columns:
                            alert[label_col] = row[label_col]
                    alerts.append(alert)
                    break
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

def is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts):
    """현재 시그니처가 알려진 FP 시그니처 중 하나의 슈퍼셋인지 확인"""
    if not known_fp_sig_dicts or not isinstance(known_fp_sig_dicts, list):
        return False # 알려진 FP 없으면 False
    if not current_sig_dict or not isinstance(current_sig_dict, dict):
        return False # 현재 시그니처 없으면 False

    for fp_sig_dict in known_fp_sig_dicts:
        if not isinstance(fp_sig_dict, dict): continue # FP 시그니처 포맷 오류 시 건너뛰기

        # fp_sig_dict의 모든 항목(키, 값)이 current_sig_dict에 존재하는지 확인
        is_superset = all(
            item in current_sig_dict.items() for item in fp_sig_dict.items()
        )
        if is_superset and len(current_sig_dict) > len(fp_sig_dict): # 진부분집합 방지 (옵션)
             # print(f"Debug: {current_sig_dict} is superset of {fp_sig_dict}")
             return True
    return False

def evaluate_false_positives(
        alerts_df: pd.DataFrame,
        current_signatures_map: dict, # 현재 시그니처 ID -> dict 맵
        known_fp_sig_dicts: list = None, # 알려진 FP 시그니처 dict 리스트
        belief_threshold: float = 0.5, # 기본 FP 임계값
        superset_strictness: float = 0.9, # 슈퍼셋일 경우 엄격도 (90%)
        attack_free_df: pd.DataFrame = None # attack_free_df 파라미터 추가 (calculate_fp_scores용)
    ):
    """
    FP 점수 계산 및 슈퍼셋 로직을 적용하여 최종 FP 판정.
    Args:
        alerts_df: 현재 알림 데이터
        current_signatures_map: {'SIG_1': {...}, 'SIG_2': {...}} 형태의 맵
        known_fp_sig_dicts: [{'DstPort': 80}, ...] 형태의 리스트
        belief_threshold: FP 판단 기준값
        superset_strictness: 슈퍼셋일 때 적용할 임계값 비율 (예: 0.9)
        attack_free_df: UFP 계산용 attack free 알림 데이터
    """
    if attack_free_df is None:
         # attack_free_df가 없으면 UFP 계산 불가 -> 임시 DataFrame 또는 에러 처리
         print("Warning: attack_free_df not provided for UFP calculation in evaluate_false_positives.")
         attack_free_df = pd.DataFrame(columns=alerts_df.columns) # 빈 DataFrame 사용 (UFP 점수 0됨)
         # 또는 raise ValueError("attack_free_df is required for UFP calculation.")

    # 1. 기본 FP 점수 계산 (NRA, HAF, UFP, belief)
    fp_scores_df = calculate_fp_scores(alerts_df, attack_free_df) # attack_free_df 전달

    # 결과 저장을 위한 초기화
    fp_results = fp_scores_df.copy()
    fp_results['is_superset'] = False
    fp_results['applied_threshold'] = belief_threshold
    fp_results['likely_false_positive'] = False

    # 알려진 FP 리스트가 없으면 초기화
    if known_fp_sig_dicts is None:
        known_fp_sig_dicts = []

    # 2. 각 시그니처에 대해 슈퍼셋 검사 및 최종 FP 판정
    for index, row in fp_results.iterrows():
        sig_id = row['signature_id']
        belief_score = row['belief']

        # 현재 시그니처 딕셔너리 가져오기
        current_sig_dict = current_signatures_map.get(sig_id)
        if not current_sig_dict:
            # print(f"Warning: Signature dictionary not found for {sig_id} in current_signatures_map.")
            continue # 맵에 없으면 건너뛰기

        # 슈퍼셋 검사
        is_super = is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts)
        fp_results.loc[index, 'is_superset'] = is_super

        # 임계값 적용
        threshold = belief_threshold
        if is_super:
            threshold = belief_threshold * superset_strictness
            fp_results.loc[index, 'applied_threshold'] = threshold

        # 최종 FP 판정
        fp_results.loc[index, 'likely_false_positive'] = belief_score < threshold

    # 최종 결과 DataFrame 반환 (요약 전)
    return fp_results[['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive']]

def summarize_fp_results(detailed_fp_results: pd.DataFrame):
     """ 그룹별로 FP 판정 결과 요약 """
     if detailed_fp_results.empty:
          return pd.DataFrame()

     summary = detailed_fp_results.groupby(['signature_id', 'signature_name']).agg(
         alerts_count=('likely_false_positive', 'size'), # 총 알림 수 (그룹 크기)
         likely_fp_count=('likely_false_positive', 'sum'), # FP로 판정된 알림 수
         avg_belief=('belief', 'mean'),
         is_superset=('is_superset', 'first'), # 슈퍼셋 여부 (어차피 시그니처별로 동일)
         applied_threshold=('applied_threshold', 'first') # 적용된 임계값
     ).reset_index()

     summary['likely_fp_rate'] = summary['likely_fp_count'] / summary['alerts_count']
     summary['final_likely_fp'] = summary['likely_fp_rate'] > 0.5 # 예시: 절반 이상 FP면 최종 FP (조정 가능)

     return summary[['signature_id', 'signature_name', 'alerts_count', 'likely_fp_count', 'likely_fp_rate', 'avg_belief', 'is_superset', 'applied_threshold', 'final_likely_fp']]
