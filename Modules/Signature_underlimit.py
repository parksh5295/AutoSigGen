# Functions for lower bounding the precision and number of signatures


def count_limit(signature_dict, count):
    # Sort by 'Precision' value in descending order and select the top 'count'
    top_n_sig = sorted(signature_dict, key=lambda x: x["Precision"], reverse=True)[:count]
    return top_n_sig

def precision_limit(signature_dict, precision_lim):
    # Leave only dictionaries with a Precision value of precision_lim or higher
    upper_sig = [d for d in signature_dict if d["Precision"] >= precision_lim]
    return upper_sig


def under_limit(signature_dict, count, precision_lim):
    """
    1. select the top count signatures with count_limit
    2. filter out the selected signatures with precision_lim or more
    3. include all signatures with precision equal to or higher than the precision value of the last signature selected with count_limit
       precision value of the last signature selected with count_limit
    """
    if not signature_dict:
        return []

    # Select the top counts
    top_signatures = count_limit(signature_dict, count)
    
    # Filter selected signatures that are precision_lim or higher
    filtered_signatures = precision_limit(top_signatures, precision_lim)
    
    if not filtered_signatures:  # Returns an empty list if no signatures exceed precision_lim
        return []
        
    # Check the precision value of the last signature selected with count_limit
    last_selected_precision = top_signatures[-1]["Precision"]
    
    # Get all signatures with precision equal to or higher than the last selected signature's precision
    all_qualified_signatures = [sig for sig in signature_dict 
                              if sig["Precision"] >= max(last_selected_precision, precision_lim)]
    
    return all_qualified_signatures