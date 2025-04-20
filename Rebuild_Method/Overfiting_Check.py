import pandas as pd
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset


def compute_fp_tp(df):
    label_column = None
    possible_label_columns = ['original_label', 'label', 'class', 'Class']

    for col in possible_label_columns:
        if col in df.columns:
            label_column = col
            break

    if label_column is None:
        print("compute_fp_tp error: label column not found in alerts_df.", df.columns.tolist())
        raise KeyError("No suitable label column found in alerts data for compute_fp_tp. Expected one of: " + str(possible_label_columns))

    normal_values = ['normal', '0', 'normal.', 'Normal', 0] # Includes the number 0

    try:
        # Check and compare label column types (string/number handling)
        if pd.api.types.is_numeric_dtype(df[label_column]):
             is_normal = df[label_column].isin([v for v in normal_values if isinstance(v, (int, float))])
        else:
             # Processing with strings
             is_normal = df[label_column].astype(str).str.lower().isin([str(v).lower() for v in normal_values])
    except Exception as e:
        print(f"Warning: Error comparing labels ‘{label_column}’. Attempted string comparison. Error: {e}")
        is_normal = df[label_column].astype(str).isin([str(v) for v in normal_values])

    tp = len(df[~is_normal]) # Notifications for real-world anomalous data
    fp = len(df[is_normal])  # Notifications for real-world normal data

    print(f"compute_fp_tp results: TP={tp}, FP={fp} (label: '{label_column}')")
    return tp, fp


def evaluate_signature_overfitting(df: pd.DataFrame, signatures: list):
    """
    Evaluate signature overfitting using the entire dataset.
    """
    print(f"\nOverfitting check started - input data shape: {df.shape}")
    print(f"Input data columns: {df.columns.tolist()}")

    alerts = []
    
    # Ensure label column exists in input df
    label_col = None
    for col in ['label', 'class', 'Class']:
        if col in df.columns:
            label_col = col
            print(f"Found label column '{label_col}' in input data.")
            break
    if label_col is None:
        print("Error: Label column not found in input DataFrame 'df' for overfitting check.")
        raise KeyError(f"Label column ({['label', 'class', 'Class']}) not found in input DataFrame.")
    
    print(f"Generating alerts for {len(signatures)} signatures...")
    for i, row in df.iterrows():
        for sig_idx, sig in enumerate(signatures):
            # signature_name에서 Signature_dict 추출
            sig_dict = sig.get('Signature_dict', sig)  # Handle different signature formats
            
            match = all(
                k in row and row[k] == v 
                for k, v in sig_dict.items()
            )
            
            if match:
                alerts.append({
                    'timestamp': pd.Timestamp.now(), # Dummy timestamp
                    'signature_id': f"SIG_{sig_idx}",
                    'original_label': row[label_col]  # *** 여기가 중요: 원본 레이블 포함 ***
                })
                break  # Apply only one signature per row
    print(f"Alert generation complete. Total alerts: {len(alerts)}")
    
    alerts_df = pd.DataFrame(alerts)
    
    if alerts_df.empty:
        print("No alerts generated during overfitting check.")
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # Check if 'original_label' is included
    if 'original_label' not in alerts_df.columns:
         print("Fatal error: 'original_label' column not found in generated alerts_df!")
         raise ValueError("'original_label' not found in alerts_df.")
    
    # Call compute_fp_tp here
    tp, fp = compute_fp_tp(alerts_df) 
    
    # Calculate FN, TN based on original DataFrame 'df'
    # (Assumes normal=0, non-normal=0. Modify if needed)
    try:
        # Compare based on label type
        if pd.api.types.is_numeric_dtype(df[label_col]):
            total_positives_actual = len(df[df[label_col] != 0])
            total_negatives_actual = len(df[df[label_col] == 0])
        else:
            # Process string 'normal' etc. (convert to lowercase)
            normal_str_values = {str(v).lower() for v in normal_values}
            is_actual_normal = df[label_col].astype(str).str.lower().isin(normal_str_values)
            total_positives_actual = len(df[~is_actual_normal])
            total_negatives_actual = len(df[is_actual_normal])
    except Exception as e:
         print(f"Error calculating actual positives/negatives: {e}")
         total_positives_actual = 0
         total_negatives_actual = len(df) # Fallback

    fn = max(0, total_positives_actual - tp) # False Negatives
    tn = max(0, total_negatives_actual - fp) # True Negatives

    # Calculate Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Overfitting check results: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    metrics_results = {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1
    }
    
    return metrics_results


def print_signature_overfit_report(results):
    print("Overfitting Report:")
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key.capitalize()}: {value:.4f}")
            else:
                print(f"  {key.capitalize()}: {value}")
    elif isinstance(results, pd.DataFrame):
         print(results.to_string())
    else:
        print(results)

# Example usage (assuming train_alerts and test_alerts are available DataFrames)
# report = evaluate_signature_overfitting(train_alerts, test_alerts)
# print_signature_overfit_report(report, signature_names={9001: 'ICMP Zero Bytes', 9002: 'Low SYN Packet'})

def evaluate_single_signature(df, signature):
    """
    Apply a single signature to a dataframe and return matching rows
    """
    conditions = []
    for feature, value in signature.items():
        if isinstance(value, (list, tuple)):
            conditions.append(df[feature].between(value[0], value[1]))
        else:
            conditions.append(df[feature] == value)
    
    return pd.concat(conditions, axis=1).all(axis=1)
