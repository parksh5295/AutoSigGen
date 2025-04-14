# Difference from Apriori: Calculate conditional probabilities directly, without looking for frequent patterns first
# It is characterized by its lack of support: there is no mechanism to calculate how much of the total a feature should appear in the calculation.

# input 'df': Embedded dataframes
# return: list of association rules that satisfy the condition List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]
from collections import defaultdict


def conditional_probability(df, min_confidence=0.8):
    # Save results in a set (use set instead of list for duplicate removal)
    rule_set = set()
    
    # Calculate value counts for each feature
    value_counts = {}
    for feature in df.columns:
        value_counts[feature] = df[feature].value_counts()
    
    # Calculate conditional probabilities for each feature pair
    for feature_x in df.columns:
        # Calculate indices for each value of feature_x
        x_value_indices = defaultdict(set)
        for idx, value in enumerate(df[feature_x]):
            x_value_indices[value].add(idx)
            
        for value_x in value_counts[feature_x].index:
            # Indices of rows with value_x
            subset_indices = x_value_indices[value_x]
            total_count = len(subset_indices)
            
            if total_count == 0:
                continue
                
            for feature_y in df.columns:
                if feature_x == feature_y:
                    continue
                
                # Calculate efficient value counts of feature_y in subset with value_x
                y_value_counts = defaultdict(int)
                for idx in subset_indices:
                    y_value_counts[df.iloc[idx][feature_y]] += 1
                
                # Calculate conditional probabilities and add rules
                for value_y, count_y in y_value_counts.items():
                    confidence = count_y / total_count
                    
                    if confidence >= min_confidence:
                        # Convert to tuple and save in set (hashable type)
                        rule_tuple = tuple(sorted([
                            (feature_x, float(value_x) if isinstance(value_x, (int, float)) else value_x),
                            (feature_y, float(value_y) if isinstance(value_y, (int, float)) else value_y)
                        ]))
                        rule_set.add(rule_tuple)
    
    # Convert final result to dictionary list
    return [dict(rule) for rule in rule_set]

