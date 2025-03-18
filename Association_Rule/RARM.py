# Algorithm: RARM (Rapid Association Rule Mining)
# Improve speed by reducing the search space by eliminating unnecessary candidates, and reduce memory usage by minimizing the set of intermediate candidates
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


# Find association rules that consider Support and Confidence with the RARM method
def rarm(df, min_support=0.5, min_confidence=0.8):
    # Data transformation: convert each row to a set
    transaction_list = [set((f"{col}={row[idx]}" for idx, col in enumerate(df.columns))) for row in df.itertuples(index=False, name=None)]
    
    # Calculating Support for individual items
    item_support = {}
    for transaction in transaction_list:
        for item in transaction:
            if item not in item_support:
                item_support[item] = 0
            item_support[item] += 1

    # Filter single items that meet the Support criteria
    num_transactions = len(transaction_list)
    valid_items = {item for item, count in item_support.items() if count / num_transactions >= min_support}
    
    # Create Candidate Itemsets and Calculate Support
    candidate_itemsets = []
    for r in range(2, len(valid_items) + 1):  # Only consider combinations of two or more
        for subset in itertools.combinations(valid_items, r):
            itemset = set(subset)
            support = get_support(transaction_list, itemset)
            if support >= min_support:
                candidate_itemsets.append(itemset)
    
    # Create association rules by applying confidence criteria
    rules = set()  # Using set to prevent duplicates
    for itemset in candidate_itemsets:
        subsets = list(itertools.chain.from_iterable(itertools.combinations(itemset, i) for i in range(1, len(itemset))))  # Create all subsets
        valid_subsets = [set(sub) for sub in subsets if get_confidence(transaction_list, set(sub), itemset) >= min_confidence]

        for valid_set in valid_subsets:
            rule_dict = {pair.split('=')[0]: int(pair.split('=')[1]) for pair in valid_set}
            sorted_rule = tuple(sorted(rule_dict.items()))  # Sorting the items to avoid duplicates based on key order
            rules.add(sorted_rule)  # Store the sorted rule as a tuple to avoid duplicate sets

    return [dict(rule) for rule in rules]  # Convert back to dictionary form

