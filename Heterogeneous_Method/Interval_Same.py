# Same interval length
# Output file: A dataframe separated by groups, with group names substituted for feature values
# The output is a dataframe and feature list divided into groups.

import pandas as pd
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues, Heterogeneous_Feature_named_combine
from utils.separate_bin import interval_length_same


def Heterogeneous_Interval_Same(data, file_type):
    categorical_features, time_features, packet_length_features, count_features, binary_features = Heterogeneous_Feature_named_featrues(file_type)
    feature_list = [categorical_features, time_features, packet_length_features, count_features, binary_features]

    data_list = Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data)

    for i in range(len(data_list)):
        data_semilist = data_list[i]
        feature_semilist = feature_list[i]
        small_df = interval_length_same(data_semilist, feature_semilist)
        df = pd.concat([df, small_df], ignore_index=True)
    return df, feature_list