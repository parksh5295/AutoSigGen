def anomal_class_data(data):
    anomal_rows = data[data['label'] == 1]
    return{
        anomal_rows
    }

def nomal_class_data(data):
    nomal_rows = data[data['label'] == 0]
    return{
        nomal_rows
    }

def without_label(data):
    data = data.columns.difference(['label'])
    return