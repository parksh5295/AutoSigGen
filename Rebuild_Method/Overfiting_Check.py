import pandas as pd
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset


def evaluate_signature_overfitting(data_df, signatures):
    """
    Evaluate whether a signature is overfitted based on its performance
    across training and testing datasets.
    """
    def compute_fp_tp(df):
        # NSL-KDD can use ‘label’ or ‘class’ columns
        label_column = None
        possible_label_columns = ['original_label', 'label', 'class', 'Class']
        
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            print("Available columns:", df.columns.tolist())
            raise KeyError("No label column found in the dataset. Expected one of: " + str(possible_label_columns))
        
        # NSL-KDD의 경우 'normal'이 아닌 다른 값(예: '0' 또는 'normal.')일 수 있음
        normal_values = ['normal', '0', 'normal.', 'Normal', 'benign', 'BENIGN']
        tp = len(df[~df[label_column].isin(normal_values)])
        fp = len(df[df[label_column].isin(normal_values)])
        
        return tp, fp

    report = {}
    
    # Perform an evaluation for each signature
    for idx, signature in enumerate(signatures, 1):
        sig_name = f"Signature_{idx}"
        
        # Apply a signature to the dataset
        formatted_sig = {
            'id': f'SIG_{idx}',
            'name': sig_name,
            'condition': lambda row, sig=signature: all(
                row[k] == v for k, v in sig.items()
            )
        }
        
        # Create alerts
        alerts = apply_signatures_to_dataset(data_df, [formatted_sig])
        
        # Calculate performance metrics
        metrics = compute_fp_tp(alerts)
        
        report[sig_name] = {
            'metrics': metrics,
            'overfit_risk': metrics[1] / (metrics[0] + metrics[1]) > 0.2  # If FPR is 20% or more, consider it overfitting
        }

    return report


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
