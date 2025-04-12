# Return needs to be received as 'data[label]='


def anomal_judgment_nonlabel(data_type, data):
    if data_type == "MiraiBotnet":
        data_line = ['reconnaissance', 'infection', 'action']
    elif data_type in ['NSL-KDD', 'NSL_KDD']:
        data_line = ['class']
    '''
    Need more setting for another data type
    '''
    
    result = data[data_line].any(axis=1).astype(int)
    # data_line: A collection of features for determining the label of nonlabel data  / e.i. ['reconnaissance', 'infection', 'action']

    return result
    

def anomal_judment_label(data):
    if data['Label']:
        return data['Label']
    elif data['label']:
        return data['label']
    else:
        print("label data error!")
        return