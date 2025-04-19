import pandas as pd
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset


def evaluate_signature_overfitting(data_df, signatures):
    """
    Evaluate signature overfitting using the entire dataset.
    """
    alerts = []
    
    # Ensure label column exists in input df
    label_col = None
    for col in ['label', 'class', 'Class']:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        print("Available columns in df for overfitting check:", df.columns.tolist())
        raise KeyError("No label column found in the input DataFrame for overfitting check.")
    
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
    
    alerts_df = pd.DataFrame(alerts)
    
    if alerts_df.empty:
        print("No alerts generated during overfitting check.")
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # Check if 'original_label' is included
    if 'original_label' not in alerts_df.columns:
         print("Error: 'original_label' not found in alerts_df within evaluate_signature_overfitting")
         print("Alerts columns:", alerts_df.columns.tolist())
         # Temporary solution to add label (should be included above)
         alerts_df['original_label'] = 'unknown' 
    
    # Call compute_fp_tp here
    metrics = compute_fp_tp(alerts_df) 
    
    # Return performance metrics
    # ... (existing return logic) ...
    return metrics # Example, actual return value may differ


def print_signature_overfit_report(report, signature_names=None):
    print("Signature Overfitting Evaluation Report")
    print("=" * 60)
    for sig_id, metrics in report.items():
        name = signature_names.get(sig_id, f"Signature {sig_id}") if signature_names else f"Signature {sig_id}"
        print(f"{name}:")
        print(f"  TPR: {metrics['metrics'][0] / (metrics['metrics'][0] + metrics['metrics'][1]):.2f}, FPR: {metrics['metrics'][1] / (metrics['metrics'][0] + metrics['metrics'][1]):.2f}")
        print(f"  Overfit Risk: {'YES' if metrics['overfit_risk'] else 'NO'}")
        print("-" * 60)

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
