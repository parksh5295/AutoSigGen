# Algorithm: SaM (Split and Merge)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools


# Calculate Support for a given itemset
def get_support(transaction_list, itemset):
    count = sum(1 for transaction in transaction_list if itemset.issubset(transaction))
    return count / len(transaction_list)


# Calculate Confidence for a given rule
def get_confidence(transaction_list, base, full):
    base_support = get_support(transaction_list, base)
    full_support = get_support(transaction_list, full)
    return full_support / base_support if base_support > 0 else 0


# SaM Algorithm for Frequent Itemset Mining
def sam(df, min_support=0.5, min_confidence=0.8):
    # Convert data into transaction format
    transaction_list = [set(f"{col}={row[idx]}" for idx, col in enumerate(df.columns)) for row in df.itertuples(index=False, name=None)]

    # Split: Divide transactions into chunks (simulating real SaM behavior)
    chunk_size = max(1, len(transaction_list) // 2)  # Split into two parts
    chunks = [transaction_list[i : i + chunk_size] for i in range(0, len(transaction_list), chunk_size)]

    # Step 1: Mine frequent patterns in each chunk
    local_frequent_itemsets = []
    for chunk in chunks:
        item_support = {}
        for transaction in chunk:
            for item in transaction:
                if item not in item_support:
                    item_support[item] = 0
                item_support[item] += 1

        num_transactions = len(chunk)
        valid_items = {item for item, count in item_support.items() if count / num_transactions >= min_support}

        # Generate frequent itemsets locally
        for r in range(2, len(valid_items) + 1):
            for subset in itertools.combinations(valid_items, r):
                itemset = set(subset)
                support = get_support(chunk, itemset)
                if support >= min_support:
                    local_frequent_itemsets.append(itemset)

    # Step 2: Merge frequent patterns
    global_frequent_itemsets = []
    for itemset in local_frequent_itemsets:
        if get_support(transaction_list, itemset) >= min_support:
            global_frequent_itemsets.append(itemset)

    # Step 3: Generate association rules with confidence check
    rules = []
    for itemset in global_frequent_itemsets:
        subsets = list(itertools.chain.from_iterable(itertools.combinations(itemset, i) for i in range(1, len(itemset))))
        valid_subsets = [set(sub) for sub in subsets if get_support(transaction_list, set(sub)) >= min_support]

        for valid_set in valid_subsets:
            confidence = get_support(transaction_list, itemset) / get_support(transaction_list, valid_set)
            if confidence >= min_confidence:
                rule_dict = {pair.split("=")[0]: int(pair.split("=")[1]) for pair in valid_set}
                rules.append(rule_dict)

    # Remove duplicates by treating {a=3, b=4} and {b=4, a=3} as the same
    unique_rules = []
    for rule in rules:
        sorted_rule = {k: rule[k] for k in sorted(rule)}
        if sorted_rule not in unique_rules:
            unique_rules.append(sorted_rule)

    return unique_rules
