# FP-Growth(Frequent Pattern Growth) Algorithm
# Better than Apriori for processing large amounts of data
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import pandas as pd
from fim import fpgrowth

def FPGrowth_rule(df, min_support=0.5, min_confidence=0.8):
    """
    Fast FP-Growth using pyfim
    Input:
        df: pandas DataFrame (categorical or binary)
        min_support: float (0.0~1.0, proportion)
        min_confidence: float (0.0~1.0, proportion)
    Output:
        List of rule dictionaries [{feature1: value1, feature2: value2, ...}, ...]
    """

    # Convert DataFrame to transactions: list of lists like ['proto=tcp', 'flag=1', ...]
    transactions = df.astype(str).apply(lambda row: [f"{col}={val}" for col, val in row.items()], axis=1).tolist()

    abs_support = int(len(transactions) * min_support * 1000 / 1000)  # pyfim uses absolute integer support
    abs_conf = int(min_confidence * 100)  # pyfim uses percent confidence (0–100)

    # Run FP-Growth in rule generation mode ('r') and report antecedents + consequents ('aC')
    raw_rules = fpgrowth(transactions, supp=abs_support, conf=abs_conf, report='aC', target='r')

    rule_dicts = []
    for rule in raw_rules:
        antecedents = rule[0]
        consequents = rule[1]

        # Parse antecedents and consequents back into dictionary form
        antecedent_dict = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in antecedents}
        consequent_dict = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in consequents}

        combined_rule = {**antecedent_dict, **consequent_dict}
        sorted_rule = {k: combined_rule[k] for k in sorted(combined_rule)}

        if sorted_rule not in rule_dicts:
            rule_dicts.append(sorted_rule)

    return rule_dicts


'''
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def FPGrowth_rule(df, min_support=0.5, min_confidence=0.8, association_metric='confidence'):
    # Decide on a matrics method
    metric = association_metric

    df_encoded = pd.get_dummies(df.astype(str), prefix_sep="=") # One-Hot Encoding Conversion

    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)    # Apply the FP-Growth algorithm
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_confidence, num_itemsets=len(frequent_itemsets)) # Create association rules

    # Convert antecedents and consequents into a single dictionary
    rule_dicts = []
    for _, row in rules.iterrows():
        antecedents = {item.split("=")[0]: int(item.split("=")[1]) for item in row['antecedents']}
        consequents = {item.split("=")[0]: int(item.split("=")[1]) for item in row['consequents']}
        
        # Combine antecedents and consequents, and sort the keys to avoid duplicates
        combined_rule = {**antecedents, **consequents}
        sorted_rule = {k: combined_rule[k] for k in sorted(combined_rule)}  # Sort the rule by key
        
        if sorted_rule not in rule_dicts:
            rule_dicts.append(sorted_rule)

    return rule_dicts
'''
