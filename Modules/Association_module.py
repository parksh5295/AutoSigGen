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


def association_module(df, association_rule_choose, min_support, min_confidence, association_metric):
    association_list = []  # Initialize with empty list

    if association_rule_choose == 'conditional_probability':
        association_list = conditional_probability(df, min_confidence)
    elif association_rule_choose in ['apriori', 'Apriori']:
        association_list = Apriori_rule(df, min_support, min_confidence, association_metric)
    elif association_rule_choose in ['fpgrowth', 'FPGrowth']:
        # association_list = FPGrowth_rule(df, min_support, min_confidence)   # PyFim
        association_list = FPGrowth_rule(df, min_support, min_confidence, association_metric) # mlxtend
    elif association_rule_choose in ['Eclat', 'eclat']:
        association_list = eclat(df, min_support, min_confidence)
    elif association_rule_choose in ['rarm', 'RARM']:
        association_list = rarm(df, min_support, min_confidence)
    elif association_rule_choose in ['h_mine', 'H_mine']:
        association_list = h_mine(df, min_support, min_confidence)
    elif association_rule_choose in ['opus', 'OPUS']:
        association_list = opus(df, min_support, min_confidence)
    elif association_rule_choose in ['sam', 'SaM']:
        association_list = sam(df, min_support, min_confidence)
    else:
        print("The name of the association rule appears to be incorrect.")
        pass

    return association_list # dictionary