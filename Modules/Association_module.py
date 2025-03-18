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


def association_module(df, association_rule_choose, min_support, min_confidence):
    if association_rule_choose == 'conditional_probability':
        association_list = conditional_probability(df, min_confidence)
    elif association_rule_choose == 'apriori' or 'Apriori':
        association_list = Apriori_rule(df, min_support, min_confidence)
    elif association_rule_choose == 'fpgrowth' or 'FPGrowth':
        association_list = FPGrowth_rule(df, min_support, min_confidence)
    elif association_rule_choose == 'eclat':
        association_list = eclat(df, min_support, min_confidence)
    elif association_rule_choose == 'rarm' or 'RARM':
        association_list = rarm(df, min_support, min_confidence)
    elif association_rule_choose == 'h_mine' or 'H_mine':
        association_list = h_mine(df, min_support, min_confidence)
    elif association_rule_choose == 'opus' or 'OPUS':
        association_list = opus(df, min_support, min_confidence)
    elif association_rule_choose == 'sam' or 'SaM':
        association_list = sam(df, min_support, min_confidence)
    else:
        print("The name of the association rule appears to be incorrect.")
        pass

    return association_list