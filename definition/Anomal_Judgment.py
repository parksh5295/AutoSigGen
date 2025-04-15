# Return needs to be received as 'data[label]='


def anomal_judgment_nonlabel(data_type, data):
    if data_type == "MiraiBotnet":
        data_line = ['reconnaissance', 'infection', 'action']
    elif data_type in ['NSL-KDD', 'NSL_KDD']:
        class_columns = [col for col in data.columns if col.lower() == 'class']
        if class_columns:
            data_line = [class_columns[0]]
        else:
            raise ValueError("No 'class' or 'Class' column found in the dataset")
    '''
    Need more setting for another data type
    '''
    
    result = data[data_line].any(axis=1).astype(int)
    # data_line: A collection of features for determining the label of nonlabel data  / e.i. ['reconnaissance', 'infection', 'action']
    print("hey1: ", type(result))

    return result, data_line
    

def anomal_judgment_label(data):
    if data['Label']:
        return data['Label']
    elif data['label']:
        return data['label']
    else:
        print("label data error!")
        return
    