# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd
import numpy as np


def calculate_signature(data, signatures):
    results = []    # list for save the results
    # Inspect each row in the DataFrame for each condition dictionary
    for signature in signatures:
        matches = data[list(signature.keys())].eq(pd.Series(signature)).all(axis=1)  # Conditions are met

        # Calculate TP, FN, FP, TN
        TP = ((matches) & (data['label'] == 1)).sum()
        FN = ((~matches) & (data['label'] == 1)).sum()
        FP = ((matches) & (data['label'] == 0)).sum()
        TN = ((~matches) & (data['label'] == 0)).sum()

        # signature 자체를 저장 (Condition_N 대신)
        results.append({
            'Signature_dict': signature,  # 실제 signature 딕셔너리 저장
            'TP': TP, 
            'TN': TN, 
            'FP': FP, 
            'FN': FN
        })    

    return results


# Tools for evaluating recall in an aggregated signature collection
def calculate_signatures(data, signatures):
    # Extract only the actual signature conditions
    needed_columns = set()
    for signature in signatures:
        needed_columns.update(signature.keys())  # signature가 이미 딕셔너리이므로 직접 사용
        print("signature: ", signature)
    
    needed_columns.add('label')
    data_subset = data[list(needed_columns)]
    
    TP = FN = FP = TN = 0
    for _, row in data_subset.iterrows():
        # 각 signature와 직접 비교
        row_satisfied = any(all(row[k] == v for k, v in sig.items()) 
                          for sig in signatures)
        
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
