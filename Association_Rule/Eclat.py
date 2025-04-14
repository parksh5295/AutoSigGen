# Algorithm: Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal)
# Using set intersection operations to find infrequent items
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools


# Calculate Support for how many times a particular itemset appears in the overall data
def get_support(transaction_list, itemset):
    count = sum(1 for transaction in transaction_list if itemset.issubset(transaction))
    return count / len(transaction_list)

def get_confidence(transaction_list, base, full):
    # Confidence = P(Full | Base) = Support(Full) / Support(Base)
    base_support = get_support(transaction_list, base)
    full_support = get_support(transaction_list, full)
    return full_support / base_support if base_support > 0 else 0


'''
# Eclat Algorithm: Finding infrequent itemsets using set intersection operations
def eclat_tid(prefix, items, min_support, total_transactions, frequent_itemsets):
    while items:
        item, tid_list = items.pop()
        new_prefix = prefix.union(item)
        support = len(tid_list) / total_transactions

        if support >= min_support:
            frequent_itemsets.add(frozenset(new_prefix))

            # Generate new candidates via TID list intersection
            new_items = []
            for other_item, other_tid_list in items:
                intersection = tid_list & other_tid_list
                if len(intersection) / total_transactions >= min_support:
                    new_items.append((item.union(other_item), intersection))

            eclat_tid(new_prefix, new_items, min_support, total_transactions, frequent_itemsets)
'''
            

# Eclat Algorithm: Finding infrequent itemsets using set intersection operations
def eclat(df, min_support=0.5, confidence_threshold=0.8):
    # Convert each row into a set of items
    transaction_list = [set((f"{col}={row[idx]}" for idx, col in enumerate(df.columns))) for row in df.itertuples(index=False, name=None)]

    # Find all unique items
    itemsets = {frozenset([value]) for row in transaction_list for value in row}
    
    frequent_itemsets = set()
    rule_set = set()  # To store unique rules without duplicates
    
    # Stack for iterative processing
    stack = [(set(), list(itemsets))]
    
    while stack:
        prefix, items = stack.pop()
        
        while items:
            item = items.pop()
            new_prefix = prefix.union(item)
            support = get_support(transaction_list, new_prefix)

            if support >= min_support:
                frequent_itemsets.add(frozenset(new_prefix))
                remaining_items = [other for other in items if get_support(transaction_list, new_prefix.union(other)) >= min_support]

                # Calculate confidence for all pairs and add to rules if confidence is above threshold
                for base in itertools.combinations(new_prefix, len(new_prefix) - 1):  # Generate subsets of size |new_prefix|-1
                    base_set = set(base)
                    confidence = get_confidence(transaction_list, base_set, new_prefix)

                    if confidence >= confidence_threshold:  # Check if confidence is above the threshold
                        rule_dict = {pair.split('=')[0]: float(pair.split('=')[1]) for pair in new_prefix}
                        sorted_rule = tuple(sorted(rule_dict.items()))

                        if sorted_rule not in rule_set:
                            rule_set.add(sorted_rule)
                
                # Add the new items for further exploration
                if remaining_items:
                    stack.append((new_prefix, remaining_items))
    
    # Return the filtered rules based on confidence threshold
    return [dict(rule) for rule in rule_set]
