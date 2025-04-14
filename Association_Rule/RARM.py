# Algorithm: RARM (Rapid Association Rule Mining)
# Improve speed by reducing the search space by eliminating unnecessary candidates, and reduce memory usage by minimizing the set of intermediate candidates
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations


class RARMiner:
    def __init__(self):
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # TID list
        self.item_counts = defaultdict(int)  # Item count
        
    def add_transaction(self, tid, items):
        # Process single transaction
        self.transaction_count += 1
        for item in items:
            self.item_tids[item].add(tid)
            self.item_counts[item] += 1
    
    def get_support_from_tids(self, tids):
        # Calculate support from TID set
        return len(tids) / self.transaction_count
    
    def get_support(self, items):
        # Calculate support for itemset
        if not items:
            return 0
        # Calculate TID intersection
        common_tids = set.intersection(*(self.item_tids[item] for item in items))
        return self.get_support_from_tids(common_tids)
    
    def get_confidence(self, base_items, full_items):
        # Calculate confidence
        base_support = self.get_support(base_items)
        if base_support == 0:
            return 0
        return self.get_support(full_items) / base_support


def rarm(df, min_support=0.5, min_confidence=0.8):
    # Initialize RARM miner
    miner = RARMiner()
    
    # Convert data and build initial structure (streaming approach)
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={val}" for col, val in zip(df.columns, row))
        miner.add_transaction(tid, items)
    
    # Find frequent 1-itemset (items with minimum support)
    frequent_items = {
        item for item, count in miner.item_counts.items()
        if count / miner.transaction_count >= min_support
    }
    
    # Set for rule storage
    rule_set = set()
    
    # Process level by level (memory efficient)
    current_level = {frozenset([item]) for item in frequent_items}
    
    while current_level and len(next(iter(current_level))) < len(frequent_items):
        next_level = set()
        
        # Generate rules from current level itemsets
        for itemset in current_level:
            # Generate next level candidates (memory efficient way)
            for item in frequent_items - itemset:
                candidate = itemset | {item}
                
                # Check if all subsets are frequent (RARM optimization)
                if all(frozenset(subset) in current_level 
                      for subset in combinations(candidate, len(itemset))):
                    
                    support = miner.get_support(candidate)
                    if support >= min_support:
                        next_level.add(candidate)
                        
                        # Generate rules (memory efficient way)
                        for i in range(1, len(candidate)):
                            for antecedent in combinations(candidate, i):
                                antecedent = frozenset(antecedent)
                                consequent = candidate - antecedent
                                
                                confidence = miner.get_confidence(antecedent, candidate)
                                if confidence >= min_confidence:
                                    # Convert rule to sorted tuple
                                    rule_dict = {}
                                    for item in antecedent:
                                        key, value = item.split('=')
                                        rule_dict[key] = int(value)
                                    
                                    rule_tuple = tuple(sorted(rule_dict.items()))
                                    rule_set.add(rule_tuple)
        
        current_level = next_level
    
    # Convert results
    return [dict(rule) for rule in rule_set]

