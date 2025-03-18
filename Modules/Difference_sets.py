# Functions for getting a list of subset lists


def dict_list_difference(list1, list2):
    # Converting a dictionary to a frozenset to perform set operations
    set1 = {frozenset(d.items()) for d in list1}
    set2 = {frozenset(d.items()) for d in list2}
    
    difference = set1 - set2
    
    # Convert back to dictionary form and return as a list
    return [dict(d) for d in difference]