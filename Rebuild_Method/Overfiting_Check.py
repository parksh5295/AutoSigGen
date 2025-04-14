def evaluate_signature_overfitting(train_alerts, test_alerts):
    """
    Evaluate whether a signature is overfitted based on its performance
    across training and testing datasets.
    """
    def compute_fp_tp(df):
        total = len(df)
        tp = len(df[df['original_label'] != 'normal'])
        fp = len(df[df['original_label'] == 'normal'])
        return {
            'total': total,
            'tp': tp,
            'fp': fp,
            'tpr': tp / total if total > 0 else 0,
            'fpr': fp / total if total > 0 else 0
        }

    report = {}
    all_signatures = set(train_alerts['signature_id']) | set(test_alerts['signature_id'])

    for sig in all_signatures:
        train_df = train_alerts[train_alerts['signature_id'] == sig]
        test_df = test_alerts[test_alerts['signature_id'] == sig]

        train_metrics = compute_fp_tp(train_df)
        test_metrics = compute_fp_tp(test_df)

        delta_tpr = train_metrics['tpr'] - test_metrics['tpr']
        delta_fpr = test_metrics['fpr'] - train_metrics['fpr']

        overfit = delta_tpr > 0.2 or delta_fpr > 0.2

        report[sig] = {
            'train': train_metrics,
            'test': test_metrics,
            'delta_tpr': round(delta_tpr, 3),
            'delta_fpr': round(delta_fpr, 3),
            'overfit_risk': overfit
        }

    return report


def print_signature_overfit_report(report, signature_names=None):
    print("Signature Overfitting Evaluation Report")
    print("=" * 60)
    for sig_id, metrics in report.items():
        name = signature_names.get(sig_id, f"Signature {sig_id}") if signature_names else f"Signature {sig_id}"
        print(f"{name}:")
        print(f"  Train TPR: {metrics['train']['tpr']:.2f}, FPR: {metrics['train']['fpr']:.2f}")
        print(f"  Test  TPR: {metrics['test']['tpr']:.2f}, FPR: {metrics['test']['fpr']:.2f}")
        print(f"  Delta TPR: {metrics['delta_tpr']:.2f}, Delta FPR: {metrics['delta_fpr']:.2f}")
        print(f"  Overfit Risk: {'YES' if metrics['overfit_risk'] else 'NO'}")
        print("-" * 60)

# Example usage (assuming train_alerts and test_alerts are available DataFrames)
# report = evaluate_signature_overfitting(train_alerts, test_alerts)
# print_signature_overfit_report(report, signature_names={9001: 'ICMP Zero Bytes', 9002: 'Low SYN Packet'})
