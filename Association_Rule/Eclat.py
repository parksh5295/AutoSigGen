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


# Eclat Algorithm: Finding infrequent itemsets using set intersection operations
def eclat_recursive(prefix, items, transaction_list, min_support, frequent_itemsets):
    while items:
        item = items.pop()
        new_prefix = prefix.union(item)
        support = get_support(transaction_list, new_prefix)

        if support >= min_support:
            frequent_itemsets.add(frozenset(new_prefix))
            remaining_items = [other for other in items if get_support(transaction_list, new_prefix.union(other)) >= min_support]
            eclat_recursive(new_prefix, remaining_items, transaction_list, min_support, frequent_itemsets)


# Take a DataFrame and find frequent patterns with Eclat
def eclat(df, min_support=0.5, min_confidence=0.8):
    # Transform data: convert each row into a set
    transaction_list = [set((f"{col}={row[idx]}" for idx, col in enumerate(df.columns))) for row in df.itertuples(index=False, name=None)]
    print("1-1")
    # Find a single frequent occurrence of each item
    itemsets = {frozenset([value]) for row in transaction_list for value in row}
    print("1-2")
    frequent_itemsets = set()  # Use set instead of list
    
    # Performing an Eclat (recursion)
    eclat_recursive(set(), list(itemsets), transaction_list, min_support, frequent_itemsets)
    print("1-3")
    # Convert results to a dictionary list
    rules = []
    existing_rules = set()  # set to avoid duplicates

    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):  # See all possible combinations
                print("1-4")
                for base in itertools.combinations(itemset, i):
                    base_set = set(base)
                    full_set = itemset
                    confidence = get_confidence(transaction_list, base_set, full_set)
                    print("1-5")
                    
                    if confidence >= min_confidence:
                        rule_dict = {pair.split('=')[0]: float(pair.split('=')[1]) for pair in full_set}

                        # Always sort and save in the same order
                        sorted_rule = tuple(sorted(rule_dict.items()))
                        print("1-6")
                        if sorted_rule not in existing_rules:
                            existing_rules.add(sorted_rule)
                            rules.append(dict(sorted_rule))  # Convert back to dictionary form

    return rules
