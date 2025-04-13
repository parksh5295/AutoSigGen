# Input: dataframe (Common table shapes)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def Apriori_rule(df, min_support=0.5, min_confidence=0.8, association_metric='confidence'):  # default; min_support=0.5, min_confidence=0.8
    # Decide on a matrics method
    metric = association_metric

    df_encoded = pd.get_dummies(df.astype(str), prefix_sep="=") # One-Hot Encoding Conversion

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True) # Applying Apriori
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_confidence, num_itemsets=len(frequent_itemsets)) # Create association rules

    # Convert antecedents and consequents into a single dictionary
    rule_dicts = []
    for _, row in rules.iterrows():
        antecedents = {item.split("=")[0]: int(item.split("=")[1]) for item in row['antecedents']}
        consequents = {item.split("=")[0]: int(item.split("=")[1]) for item in row['consequents']}

        combined_rule = {**antecedents, **consequents}  # Combine two into one dictionary
        sorted_rule = {k: combined_rule[k] for k in sorted(combined_rule)}  # Sort the rule by key

        if sorted_rule not in rule_dicts:
            rule_dicts.append(sorted_rule)

    return rule_dicts
