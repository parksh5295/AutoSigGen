# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd


def calculate_signature(data, signatures):
    results = []    # list for save the results
    # Inspect each row in the DataFrame for each condition dictionary
    for idx, signature in enumerate(signatures):
        signature_name = f'Condition_{idx + 1}'
        matches = data[list(signature.keys())].eq(pd.Series(signature)).all(axis=1)  # Conditions are met

        # Calculate TP, FN, FP, TN
        TP = ((matches) & (data['label'] == 1)).sum()
        FN = ((~matches) & (data['label'] == 1)).sum()
        FP = ((matches) & (data['label'] == 0)).sum()
        TN = ((~matches) & (data['label'] == 0)).sum()

        results.append({'Signature_name': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})    

    return results


# Tools for evaluating recall in an aggregated signature collection
def calculate_signatures(data, signatures):
    # Initialize variables for storing results
    TP = FN = FP = TN = 0

    # Inspect each row in the DataFrame(data)
    for _, row in data.iterrows():
        row_satisfied = any(all(row.get(k) == v for k, v in signature.items()) for signature in signatures)
        
        if row['label'] == 1:
            if row_satisfied:
                TP += 1  # Have a dictionary that satisfies the condition
            else:
                FN += 1  # No dictionaries satisfy the condition
        else:  # row['label'] == 0
            if row_satisfied:
                FP += 1  # Have a dictionary that satisfies the condition
            else:
                TN += 1  # No dictionaries satisfy the condition

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return recall

