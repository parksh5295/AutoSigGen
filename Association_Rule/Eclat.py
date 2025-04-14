# Algorithm: Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal)
# Using set intersection operations to find infrequent items
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools
from collections import defaultdict


# Calculate Support for how many times a particular itemset appears in the overall data
def get_support(transaction_list, itemset):
    count = sum(1 for transaction in transaction_list if itemset.issubset(transaction))
    return count / len(transaction_list)

def get_confidence(transaction_list, base, full):
    # Confidence = P(Full | Base) = Support(Full) / Support(Base)
    base_support = get_support(transaction_list, base)
    full_support = get_support(transaction_list, full)
    return full_support / base_support if base_support > 0 else 0


def get_support_optimized(tid_map, itemset):
    # Optimize support calculation using tid_map
    if len(itemset) == 1:
        return len(tid_map[next(iter(itemset))]) / tid_map['total']
    
    # Calculate support through intersection calculation
    tids = set.intersection(*[tid_map[item] for item in itemset])
    return len(tids) / tid_map['total']


def get_confidence_optimized(tid_map, base, full):
    base_support = get_support_optimized(tid_map, base)
    full_support = get_support_optimized(tid_map, full)
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
    # Create TID mapping (for memory and time efficiency)
    tid_map = defaultdict(set)
    tid_map['total'] = len(df)
    
    # Calculate TID list for each item (for memory and time efficiency)
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        for col_idx, col in enumerate(df.columns):
            item = f"{col}={row[col_idx]}"
            tid_map[item].add(tid)
    
    # Create initial 1-itemsets
    itemsets = {frozenset([item]) for item in tid_map.keys() if item != 'total'}
    
    frequent_itemsets = set()
    rule_set = set()
    
    # Stack of items to process
    stack = [(set(), list(itemsets))]
    
    while stack:
        prefix, items = stack.pop()
        
        while items:
            item = items.pop()
            new_prefix = prefix.union(item)
            support = get_support_optimized(tid_map, new_prefix)
            
            if support >= min_support:
                frequent_itemsets.add(frozenset(new_prefix))
                
                # Filter remaining items that meet support condition
                remaining_items = []
                for other in items:
                    combined = new_prefix.union(other)
                    if get_support_optimized(tid_map, combined) >= min_support:
                        remaining_items.append(other)
                
                # Calculate confidence and generate rules
                for base_size in range(1, len(new_prefix)):
                    for base in itertools.combinations(new_prefix, base_size):
                        base_set = set(base)
                        confidence = get_confidence_optimized(tid_map, base_set, new_prefix)
                        
                        if confidence >= confidence_threshold:
                            rule_dict = {}
                            for pair in new_prefix:
                                key, value = pair.split('=')
                                rule_dict[key] = float(value)
                            
                            sorted_rule = tuple(sorted(rule_dict.items()))
                            rule_set.add(sorted_rule)
                
                # Add the new items for further exploration
                if remaining_items:
                    stack.append((new_prefix, remaining_items))
    
    # Return the filtered rules based on confidence threshold
    return [dict(rule) for rule in rule_set]
