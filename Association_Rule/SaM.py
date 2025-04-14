# Algorithm: SaM (Split and Merge)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations


class ChunkProcessor:
    def __init__(self, min_support):
        self.transaction_count = 0
        self.item_counts = defaultdict(int)
        self.min_support = min_support
        
    def process_transaction(self, transaction):
        # Process single transaction
        self.transaction_count += 1
        for item in transaction:
            self.item_counts[item] += 1
    
    def get_frequent_items(self):
        # Return items with minimum support
        return {item for item, count in self.item_counts.items() 
               if count / self.transaction_count >= self.min_support}


class SaMiner:
    def __init__(self, min_support, chunk_size=1000):
        self.min_support = min_support
        self.chunk_size = chunk_size
        self.transaction_count = 0
        self.item_tids = defaultdict(set)
    
    def get_support(self, items):
        # Calculate support using TID intersection
        if not items:
            return 0
        common_tids = set.intersection(*(self.item_tids[item] for item in items))
        return len(common_tids) / self.transaction_count
    
    def process_chunk(self, transactions):
        # Process chunk
        processor = ChunkProcessor(self.min_support)
        
        # Process transactions in chunk
        for transaction in transactions:
            processor.process_transaction(transaction)
        
        return processor.get_frequent_items()


def sam(df, min_support=0.5, min_confidence=0.8):
    # Initialize
    miner = SaMiner(min_support)
    chunk_size = max(1, len(df) // 4)  # Control memory usage
    
    # Build TID mapping (streaming approach)
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        transaction = set(f"{col}={val}" for col, val in zip(df.columns, row))
        miner.transaction_count += 1
        for item in transaction:
            miner.item_tids[item].add(tid)
    
    # Split step: Process chunks
    all_frequent_items = set()
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    for chunk in chunks:
        # Convert chunk to transaction form
        transactions = [
            set(f"{col}={val}" for col, val in zip(df.columns, row))
            for row in chunk.itertuples(index=False, name=None)
        ]
        
        # Find frequent items in chunk
        frequent_items = miner.process_chunk(transactions)
        all_frequent_items.update(frequent_items)
    
    # Merge step: Create global frequent itemset
    rule_set = set()
    current_level = {frozenset([item]) for item in all_frequent_items 
                    if miner.get_support({item}) >= min_support}
    
    while current_level:
        next_level = set()
        
        # Generate rules from current level itemset
        for itemset in current_level:
            # Generate next level candidates
            for item in all_frequent_items - itemset:
                candidate = itemset | {item}
                
                # Check if all subsets are frequent
                if all(frozenset(subset) in current_level 
                      for subset in combinations(candidate, len(itemset))):
                    
                    support = miner.get_support(candidate)
                    if support >= min_support:
                        next_level.add(candidate)
                        
                        # Generate rules
                        for i in range(1, len(candidate)):
                            for antecedent in combinations(candidate, i):
                                antecedent = frozenset(antecedent)
                                consequent = candidate - antecedent
                                
                                ant_support = miner.get_support(antecedent)
                                if ant_support > 0:
                                    confidence = miner.get_support(candidate) / ant_support
                                    
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
