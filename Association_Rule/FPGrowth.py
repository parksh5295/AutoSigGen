# FP-Growth(Frequent Pattern Growth) Algorithm
# Better than Apriori for processing large amounts of data
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def FPGrowth_rule(df, min_support=0.5, min_confidence=0.8):
    # Decide on a matrics method
    print("You need to decide on a metric method for your FP-Growth algorithm.")
    metric = str(input("There are 5 in total; confidence, lift, leverage, conviction, and zhangs metric: "))

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

