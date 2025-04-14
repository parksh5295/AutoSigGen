# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd
import numpy as np


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
    # 1. Extract only the necessary columns (from signature_name)
    needed_columns = set().union(*(sig.keys() for sig in signatures))
    needed_columns.add('label')
    data_subset = data[list(needed_columns)]
    
    TP = FN = FP = TN = 0
    for _, row in data_subset.iterrows():
        # signature_name이 아닌 signature 자체를 사용
        row_satisfied = any(all(row.get(k) == v for k, v in signature.items()) 
                          for signature in signatures)
        
        if row['label'] == 1:
            if row_satisfied:
                TP += 1
            else:
                FN += 1
        else:
            if row_satisfied:
                FP += 1
            else:
                TN += 1

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall
