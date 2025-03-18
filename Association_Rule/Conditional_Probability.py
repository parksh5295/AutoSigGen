# Difference from Apriori: Calculate conditional probabilities directly, without looking for frequent patterns first
# It is characterized by its lack of support: there is no mechanism to calculate how much of the total a feature should appear in the calculation.

# input 'df': Embedded dataframes
# return: list of association rules that satisfy the condition List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]


def conditional_probability(df, min_confidence=0.8): # Confidence for making association rules; default: 0.8
    association_list = []
    feature_columns = df.columns  # Only Feature list
    
    for feature_x in feature_columns:
        unique_x_values = df[feature_x].unique()  # The only values of feature_x
        
        for value_x in unique_x_values:
            subset = df[df[feature_x] == value_x]  # Subset with a specific value (value_x)
            total_count = len(subset)  # Total Count
            
            for feature_y in feature_columns:
                if feature_x == feature_y:
                    continue
                
                unique_y_values = subset[feature_y].unique()  # The only values of feature_y
                
                for value_y in unique_y_values:
                    count_y = (subset[feature_y] == value_y).sum()  # Number of occurrences of a specific value (value_y)
                    confidence = count_y / total_count  # Calculate conditional probabilities
                    
                    if confidence >= min_confidence:
                        rule = {feature_x: value_x, feature_y: value_y}
                        # Sort the rule by feature name to avoid duplicates like {a=3, b=4} and {b=4, a=3}
                        sorted_rule = {k: rule[k] for k in sorted(rule)}
                        if sorted_rule not in association_list:
                            association_list.append(sorted_rule)
    
    return association_list

