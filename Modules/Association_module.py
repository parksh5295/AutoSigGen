# Modules to help you determine association rules
# Return: A list containing dictionaries with feature names and corresponding group numbers (each dictionary will be a signature).

from Association_Rule.Conditional_Probability import conditional_probability
from Association_Rule.Apriori import Apriori_rule
from Association_Rule.FPGrowth import FPGrowth_rule
from Association_Rule.Eclat import eclat
from Association_Rule.RARM import rarm
from Association_Rule.H_mine import h_mine
from Association_Rule.OPUS import opus
from Association_Rule.SaM import sam
import time


def association_module(df, association_rule_choose, min_support, min_confidence, association_metric):
    association_list = []
    print(f"  [Debug] >>> Entering association_module for algorithm: {association_rule_choose}") # 진입 로그
    print(f"  [Debug]     Input data shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}") # 입력 정보 로그
    start_time = time.time() # Start measuring time

    # Add logs before calling each algorithm
    if association_rule_choose == 'conditional_probability':
        print(f"  [Debug]     Calling conditional_probability function...")
        association_list = conditional_probability(df, min_confidence)
    elif association_rule_choose in ['apriori', 'Apriori']:
        print(f"  [Debug]     Calling Apriori_rule function...")
        association_list = Apriori_rule(df, min_support, min_confidence, association_metric)
    elif association_rule_choose in ['fpgrowth', 'FPGrowth']:
        # association_list = FPGrowth_rule(df, min_support, min_confidence)   # PyFim
        print(f"  [Debug]     Calling FPGrowth_rule function...")
        association_list = FPGrowth_rule(df, min_support, min_confidence, association_metric) # mlxtend
    elif association_rule_choose in ['Eclat', 'eclat']:
        print(f"  [Debug]     Calling eclat function...")
        association_list = eclat(df, min_support, min_confidence)
    elif association_rule_choose in ['rarm', 'RARM']:
        print(f"  [Debug]     Calling rarm function...")
        association_list = rarm(df, min_support, min_confidence)
    elif association_rule_choose in ['h_mine', 'H_mine']:
        print(f"  [Debug]     Calling h_mine function...")
        association_list = h_mine(df, min_support, min_confidence)
    elif association_rule_choose in ['opus', 'OPUS']:
        print(f"  [Debug]     Calling opus function...")
        association_list = opus(df, min_support, min_confidence)
    elif association_rule_choose in ['sam', 'SaM']:
        print(f"  [Debug]     Calling sam function...")
        association_list = sam(df, min_support, min_confidence)
    else:
        print("The name of the association rule appears to be incorrect.")
        pass

    end_time = time.time() # End time measurement
    print(f"  [Debug] <<< Finished {association_rule_choose}. Found {len(association_list)} rules. Time taken: {end_time - start_time:.2f} seconds.") # 종료 및 시간 로그

    return association_list # dictionary