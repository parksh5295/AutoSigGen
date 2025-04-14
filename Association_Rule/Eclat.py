# Algorithm: Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal)
# Using set intersection operations to find infrequent items
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools


'''
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


def eclat(df, min_support=0.5):
    transaction_list = [set(f"{col}={row[idx]}" for idx, col in enumerate(df.columns))
                        for row in df.itertuples(index=False, name=None)]
    total_transactions = len(transaction_list)
    print("1-1")

    # Step 1: Create initial TID lists
    tid_dict = {}
    for tid, transaction in enumerate(transaction_list):
        for item in transaction:
            key = frozenset([item])
            if key not in tid_dict:
                tid_dict[key] = set()
            tid_dict[key].add(tid)
    print("1-2")

    # Step 2: Filter by min_support
    items = [(item, tids) for item, tids in tid_dict.items()
             if len(tids) / total_transactions >= min_support]
    print("1-3")

    # Step 3: Run Eclat with TID list
    frequent_itemsets = set()
    eclat_tid(set(), items, min_support, total_transactions, frequent_itemsets)
    print("1-4")

    # Step 4: Convert back to rule format
    rules = []
    for itemset in frequent_itemsets:
        rule_dict = {pair.split('=')[0]: float(pair.split('=')[1]) for pair in itemset}
        rules.append(rule_dict)
    print("1-5")

    return rules
