# Algorithm: H-Mine (H-Structure Mining)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations


class HStructure:
    def __init__(self):
        self.item_counts = defaultdict(int)
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # Save transaction IDs where each item appears
    
    def add_transaction(self, tid, items):
        self.transaction_count += 1
        for item in items:
            self.item_counts[item] += 1
            self.item_tids[item].add(tid)
    
    def get_support(self, items):
        if not items:
            return 0
        # Calculate the number of transactions where all items appear simultaneously
        common_tids = set.intersection(*[self.item_tids[item] for item in items])
        return len(common_tids) / self.transaction_count


def h_mine(df, min_support=0.5, min_confidence=0.8):
    # Initialize H-Structure
    h_struct = HStructure()
    
    # Convert data and build H-Structure
    transaction_items = []
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={row[idx]}" for idx, col in enumerate(df.columns))
        transaction_items.append(items)  # Keep full transactions for later use
        h_struct.add_transaction(tid, items)
    
    # Find frequent 1-itemsets
    frequent_items = {
        item for item, count in h_struct.item_counts.items()
        if count / h_struct.transaction_count >= min_support
    }
    
    # Use set for rule storage (optimized for duplicate removal)
    rule_set = set()
    
    # Generate frequent itemsets and extract rules
    current_level = [frozenset([item]) for item in frequent_items]
    
    while current_level:
        next_level = set()
        
        # Generate rules from current level itemsets
        for itemset in current_level:
            # Generate rules for subsets
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Calculate confidence
                    ant_support = h_struct.get_support(antecedent)
                    if ant_support > 0:
                        confidence = h_struct.get_support(itemset) / ant_support
                        
                        if confidence >= min_confidence:
                            # Convert rule to sorted tuple and save
                            rule_dict = {}
                            for item in itemset:
                                key, value = item.split('=')
                                rule_dict[key] = int(value)
                            
                            rule_tuple = tuple(sorted(rule_dict.items()))
                            rule_set.add(rule_tuple)
            
            # Generate candidate itemsets for next level
            for other in current_level:
                if len(other) == len(itemset):
                    new_itemset = itemset.union(other)
                    if len(new_itemset) == len(itemset) + 1:
                        # Check if all subsets are frequent
                        all_frequent = True
                        for subset in combinations(new_itemset, len(itemset)):
                            if frozenset(subset) not in current_level:
                                all_frequent = False
                                break
                        
                        if all_frequent and h_struct.get_support(new_itemset) >= min_support:
                            next_level.add(frozenset(new_itemset))
        
        current_level = next_level
    
    # Convert final result to dictionary list
    return [dict(rule) for rule in rule_set]
