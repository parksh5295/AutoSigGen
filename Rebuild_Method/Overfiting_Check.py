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


def evaluate_signature_overfitting(total_signatures_count: int, high_fp_signatures_count: int):
    """
    Calculate overfitting score based on the ratio of high FP signatures to total signatures.
    No label usage.

    Args:
        total_signatures_count: The total number of signatures evaluated.
        high_fp_signatures_count: The number of signatures identified as 'high FP' in the FalsePositive_Check step.

    Returns:
        dict: Overfitting-related information (score, total count, FP count)
    """
    print(f"\nOverfitting score calculation started:")
    print(f"  - Total signatures count: {total_signatures_count}")
    print(f"  - High FP signatures count: {high_fp_signatures_count}")

    if total_signatures_count <= 0:
        print("  - Total signatures count is 0, so score calculation is not possible.")
        overfitting_score = 0.0 # or None or NaN
    else:
        # Calculate overfitting score (high FP ratio)
        overfitting_score = high_fp_signatures_count / total_signatures_count
        print(f"  - Calculated overfitting score (high FP ratio): {overfitting_score:.4f}")

    results = {
        'overfitting_score': overfitting_score,
        'total_signatures': total_signatures_count,
        'high_fp_signatures': high_fp_signatures_count
    }
    return results


def print_signature_overfit_report(results):
    """
    Print the new overfitting score results in the specified format.
    """
    print("Overfitting Score Report:")
    if isinstance(results, dict):
        score = results.get('overfitting_score', 'N/A')
        total_sigs = results.get('total_signatures', 'N/A')
        fp_sigs = results.get('high_fp_signatures', 'N/A')

        print(f"  - Overfitting Score (High FP Ratio): {score:.4f}" if isinstance(score, float) else f"  - Overfitting Score (High FP Ratio): {score}")
        print(f"  - Total Signatures Considered: {total_sigs}")
        print(f"  - High FP Signatures Identified: {fp_sigs}")
    else:
        print("  - Invalid results format provided.")

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
