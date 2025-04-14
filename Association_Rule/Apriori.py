# Input: dataframe (Common table shapes)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def Apriori_rule(df, min_support=0.5, min_confidence=0.8, association_metric='confidence'):  # default; min_support=0.5, min_confidence=0.8
    # Decide on a metrics method
    metric = association_metric

    # One-Hot Encoding Conversion - sparse=True로 메모리 효율성 향상
    df_encoded = pd.get_dummies(df.astype(str), prefix_sep="=", sparse=True)

    # Applying Apriori
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Create association rules
    rules = association_rules(frequent_itemsets, metric=metric, 
                            min_threshold=min_confidence, 
                            num_itemsets=len(frequent_itemsets))

    # Pre-split column names for faster processing
    column_map = {col: col.split("=") for col in df_encoded.columns}
    
    # Convert to set for faster duplicate checking
    rule_dicts = set()
    
    # Use to_dict('records') instead of iterrows() for faster processing
    for rule in rules[['antecedents', 'consequents']].to_dict('records'):
        combined_rule = {}
        
        # Process antecedents and consequents together
        for items in (rule['antecedents'], rule['consequents']):
            for item in items:
                key, value = column_map[item]
                combined_rule[key] = int(value)
        
        # Convert to sorted tuple for faster set addition
        rule_tuple = tuple(sorted(combined_rule.items()))
        rule_dicts.add(rule_tuple)

    # Convert set to final result format
    return [dict(rule) for rule in rule_dicts]
