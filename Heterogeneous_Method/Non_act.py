# Not embedding features separately by their attributes
# Output data is 'feature_list'
# The output is a dataframe and feature list divided into groups.
# Encoding and Normalization

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def Heterogeneous_Non_OneHotEncoder(data):
    feature = data.columns.tolist()
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encode_data = encoder.fit_transform(data[feature])
    df = pd.DataFrame(encode_data)
    return df, encode_data

def Heterogeneous_Non_StandardScaler(data):
    feature = data.columns.tolist()
    scaler_time = StandardScaler()
    encode_data = scaler_time.fit_transform(data[feature])
    df = pd.DataFrame(encode_data)
    return df, encode_data