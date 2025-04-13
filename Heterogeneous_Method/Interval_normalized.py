# Encoding and Normalization
# Section length; keep the number of sections the same for each section
# Output file: A dataframe separated by groups, with group names substituted for feature values
# The output is a dataframe and feature list divided into groups.

import pandas as pd
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues, Heterogeneous_Feature_named_combine, Heterogeneous_Feature_named_combine_standard
from utils.separate_bin import interval_length_Inverse_Count
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe


def Heterogeneous_Interval_Inverse(data, file_type, regul):
    feature_name = Heterogeneous_Feature_named_featrues(file_type)

    categorical_features = feature_name['categorical_features']
    time_features = feature_name['time_features']
    packet_length_features = feature_name['packet_length_features']
    count_features = feature_name['count_features']
    binary_features = feature_name['binary_features']

    feature_list = [categorical_features, time_features, packet_length_features, count_features, binary_features]
    print("hey: ", feature_list)

    df = pd.DataFrame() # A dataframe to store the entire condition

    if regul in ['Y', 'y']:
        data_list = Heterogeneous_Feature_named_combine_standard(categorical_features, time_features, packet_length_features, count_features, binary_features, data)
        category_mapping = False

        full_group_mapping_info = {}  # a dict to hold the mapping information for all features

        for i in range(1, len(data_list)):    # Without categorical_features
            data_semilist = data_list[i]
            feature_semilist = feature_list[i]
            small_df, group_mapping_info_small = interval_length_Inverse_Count(data_semilist, feature_semilist)
            full_group_mapping_info.update(group_mapping_info_small)    # Accumulating mapping information
            df = pd.concat([df, small_df], axis=1, ignore_index=False)   # axis=1 is for combining df per feature in column direction

    elif regul in ['N', 'n']:
        data_list, category_mapping = Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data)

        full_group_mapping_info = {}  # a dict to hold the mapping information for all features

        for i in range(1, len(data_list)-1):    # Without categorical_features, flag features
            data_semilist = data_list[i]
            feature_semilist = feature_list[i]
            # If the list is empty, skip
            if not feature_semilist:
                continue

            small_df, group_mapping_info_small = interval_length_Inverse_Count(data_semilist, feature_semilist)
            full_group_mapping_info.update(group_mapping_info_small)    # Accumulating mapping information
            df = pd.concat([df, small_df], axis=1, ignore_index=False)   # axis=1 is for combining df per feature in column direction
        df = pd.concat([df, data_list[len(data_list)-1]], axis=1, ignore_index=False)    # Adding Binary Features to DF Later

        interval_mapping_df = build_interval_mapping_dataframe(full_group_mapping_info)
        category_mapping['interval'] = interval_mapping_df  # This, in turn, calls the mapping_info
        print("category_mapping: ", category_mapping)

    df = pd.concat([data_list[0], df], axis=1, ignore_index=False)   # Adding Categorical Features to DF Later

    print("embedded data: ", df)

    return df, feature_list, category_mapping   # df = embedded data
    # category_mapping: dict; categorical, interval, binary (key)