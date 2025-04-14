# Algorithm: OPUS (Optimal Pattern Discovery)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations


class OPUSMiner:
    def __init__(self):
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # Save transaction IDs where each item appears
        self.support_cache = {}  # support value caching
    
    def add_transaction(self, tid, items):
        self.transaction_count += 1
        for item in items:
            self.item_tids[item].add(tid)
    
    def get_support(self, items):
        if not items:
            return 0
            
        # If cached support value exists, return it
        items_key = frozenset(items)
        if items_key in self.support_cache:
            return self.support_cache[items_key]
        
        # Calculate support using TID intersection
        common_tids = set.intersection(*(self.item_tids[item] for item in items))
        support = len(common_tids) / self.transaction_count
        
        # Cache support values for frequently used itemsets
        if len(items) <= 3:  # Only cache small itemsets
            self.support_cache[items_key] = support
            
        return support
    
    def prune_candidates(self, candidates, min_support):
        # Prune candidates using OPUS style
        return {c for c in candidates if self.get_support(c) >= min_support}


def opus(df, min_support=0.5, min_confidence=0.8):
    # Initialize OPUS miner
    miner = OPUSMiner()
    
    # Convert data and build initial structure
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={val}" for col, val in zip(df.columns, row))
        miner.add_transaction(tid, items)
    
    # Find frequent 1-itemsets
    frequent_items = {
        item for item in miner.item_tids
        if len(miner.item_tids[item]) / miner.transaction_count >= min_support
    }
    
    # Set for rule storage
    rule_set = set()
    
    # Incremental pattern discovery using OPUS style
    current_level = {frozenset([item]) for item in frequent_items}
    
    while current_level:
        next_candidates = set()
        
        # Generate rules from current level itemsets
        for itemset in current_level:
            # Efficient subset processing
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    ant_support = miner.get_support(antecedent)
                    if ant_support > 0:
                        confidence = miner.get_support(itemset) / ant_support
                        
                        if confidence >= min_confidence:
                            # Convert rule to sorted tuple
                            rule_dict = {}
                            for item in itemset:
                                key, value = item.split('=')
                                rule_dict[key] = int(value)
                            
                            rule_tuple = tuple(sorted(rule_dict.items()))
                            rule_set.add(rule_tuple)
            
            # Generate next level candidates
            for other in current_level:
                if len(other) == len(itemset):
                    new_candidate = itemset.union(other)
                    if len(new_candidate) == len(itemset) + 1:
                        # Check if all subsets are frequent
                        if all(frozenset(subset) in current_level 
                              for subset in combinations(new_candidate, len(itemset))):
                            next_candidates.add(new_candidate)
        
        # Prune candidates using OPUS style to determine next level
        current_level = miner.prune_candidates(next_candidates, min_support)
        
        # Memory management: Clear support cache when no longer needed
        if len(current_level) == 0:
            miner.support_cache.clear()
    
    # Convert results
    return [dict(rule) for rule in rule_set]
