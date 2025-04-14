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
    needed_columns = set().union(*(sig['signature_name'].keys() for sig in signatures))
    needed_columns.add('label')
    data_subset = data[list(needed_columns)]
    
    # 2. Use vectorized operations
    TP = FN = FP = TN = 0
    data_values = data_subset.values
    
    chunk_size = 10000
    for i in range(0, len(data_values), chunk_size):
        chunk = data_values[i:i + chunk_size]
        
        # Check matching for each signature
        matches = np.zeros(len(chunk), dtype=bool)
        for signature in signatures:
            # Extract actual conditions from signature_name
            actual_signature = signature['signature_name']
            sig_match = np.ones(len(chunk), dtype=bool)
            for k, v in actual_signature.items():
                col_idx = data_subset.columns.get_loc(k)
                sig_match &= (chunk[:, col_idx] == v)
            matches |= sig_match
        
        labels = chunk[:, data_subset.columns.get_loc('label')]
        TP += np.sum((matches) & (labels == 1))
        FN += np.sum((~matches) & (labels == 1))
        FP += np.sum((matches) & (labels == 0))
        TN += np.sum((~matches) & (labels == 0))

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall
