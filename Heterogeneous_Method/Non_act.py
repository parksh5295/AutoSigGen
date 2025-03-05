# Not embedding features separately by their attributes
# Output data is 'feature_list'

from sklearn.preprocessing import StandardScaler, OneHotEncoder


def Heterogeneous_Non_OneHotEncoder(data):
    feature = data.columns.tolist()
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encode_data = encoder.fit_transform(data[feature])
    return{
        encode_data
    }

def Heterogeneous_Non_StandardScaler(data):
    feature = data.columns.tolist()
    scaler_time = StandardScaler()
    encode_data = scaler_time.fit_transform(data[feature])
    return{
        encode_data
    }