from definition.Anomal_Judgment import anomal_judgment_nonlabel


def anomal_class_data(data):
    anomal_rows = data[data['label'] == 1]
    return anomal_rows

def nomal_class_data(data):
    nomal_rows = data[data['label'] == 0]
    return nomal_rows

def without_label(data):
    data = data.columns.difference(['label'])
    return data

def without_labelmaking_out(data_type, data):
    r, data_line = anomal_judgment_nonlabel(data_type, data)    # data_line: 'Column' name to determine label
    # r is the output of the anomalous judgment function, an argument to receive the value of data[label]. It is not used in this function.
    data = without_label(data)
    data = data.columns.difference(data[data_line])
    return data