# Return: A list of dictionaries containing the accuracy, precision, recall, F1-Score, and total_score for each signature

from Evaluation.calculate_signature import calculate_signature
from Evaluation.calculate_score import calculate_score, calculate_total_score


def signature_evaluate(data, signatures):
    signature_metrics = []
    signature_results = calculate_signature(data, signatures)
    for signature_result in signature_results:
        metric = calculate_score(signature_result['TP'], signature_result['TN'], signature_result['FP'], signature_result['FN'])
        metric['signature_name'] = signature_result
        signature_metric = metric
        weights = {'Accuracy': 0.1, 'Precision': 0.7, 'Recall': 0.15, 'F1-Score': 0.05}
        total_score = calculate_total_score(signature_metric, weights)
        signature_metric['total_score'] = total_score
        # signature_metric['total_score': total_score]
        signature_metrics.append(signature_metric)

    return signature_metrics