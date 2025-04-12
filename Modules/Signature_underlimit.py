# Functions for lower bounding the precision and number of signatures


def count_limit(signature_dict, count):
    # Sort by 'Precision' value in descending order and select the top 'count'
    top_20_sig = sorted(signature_dict, key=lambda x: x["Precision"], reverse=True)[:count]
    return top_20_sig

def precision_limit(signature_dict, precision_lim):
    # Leave only dictionaries with a Precision value of 0.8 or higher
    upper_sig = [d for d in signature_dict if d["Precision"] >= precision_lim]
    return upper_sig


def under_limit(signature_dict, count, precision_lim):
    sig1 = count_limit(signature_dict, count)
    sig2 = precision_limit(sig1, precision_lim)
    return sig2