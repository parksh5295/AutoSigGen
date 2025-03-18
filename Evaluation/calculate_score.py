# A collection of functions used to calculate scores and total points


def calculate_score(TP, TN, FP, FN):
    metrics = {}
    metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    metrics['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    metrics['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    metrics['F1-Score'] = (2 * metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) \
                          if (metrics['Precision'] + metrics['Recall']) > 0 else 0
    return metrics

# Total score calculation function
def calculate_total_score(score_dict, weights):
    total_score = sum(score_dict[key] * weights[key] for key in weights)
    return total_score


# Confusion matrix calculation functions
def calculate_confusion_matrix(eva_row, test_list, anomal):
    is_subset = set(eva_row).issubset(set(test_list))
    TP, FP, TN, FN = 0, 0, 0, 0
    if anomal == 1:  # If anomal = 1
        if is_subset:
            TP = 1
        else:
            FP = 1
    else:  # If anomal = 0
        if is_subset:
            FN = 1
        else:
            TN = 1
    return TP, FP, TN, FN