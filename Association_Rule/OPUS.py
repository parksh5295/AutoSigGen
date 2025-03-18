# Algorithm: OPUS (Optimal Pattern Discovery)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools


# Calculate Support for how many times a particular itemset appears in the overall data
def get_support(transaction_list, itemset):
    count = sum(1 for transaction in transaction_list if itemset.issubset(transaction))
    return count / len(transaction_list)


# OPUS Algorithm for Optimal Pattern Discovery
def opus(df, min_support=0.5, min_confidence=0.8):
    # Convert each row to a set of key-value pairs
    transaction_list = [set((f"{col}={row[idx]}" for idx, col in enumerate(df.columns))) for row in df.itertuples(index=False, name=None)]
    
    # Calculate support for individual items
    item_support = {}
    for transaction in transaction_list:
        for item in transaction:
            if item not in item_support:
                item_support[item] = 0
            item_support[item] += 1

    # Filter out items that do not meet the minimum support
    num_transactions = len(transaction_list)
    frequent_items = {item for item, count in item_support.items() if count / num_transactions >= min_support}

    # Generate candidate itemsets using OPUS-style pruning
    frequent_itemsets = []
    for r in range(2, len(frequent_items) + 1):
        for subset in itertools.combinations(frequent_items, r):
            itemset = set(subset)
            support = get_support(transaction_list, itemset)
            if support >= min_support:
                frequent_itemsets.append(itemset)

    # Generate association rules with confidence check
    rules = []
    for itemset in frequent_itemsets:
        subsets = list(itertools.chain.from_iterable(itertools.combinations(itemset, i) for i in range(1, len(itemset))))
        valid_subsets = [set(sub) for sub in subsets if get_support(transaction_list, set(sub)) >= min_support]

        for valid_set in valid_subsets:
            confidence = get_support(transaction_list, itemset) / get_support(transaction_list, valid_set)
            if confidence >= min_confidence:
                rule_dict = {pair.split("=")[0]: int(pair.split("=")[1]) for pair in valid_set}
                rules.append(rule_dict)

    # Sort rules to treat {a=3, b=4} and {b=4, a=3} as the same
    unique_rules = []
    for rule in rules:
        sorted_rule = {k: rule[k] for k in sorted(rule)}  # 정렬하여 중복 제거
        if sorted_rule not in unique_rules:
            unique_rules.append(sorted_rule)

    return unique_rules
