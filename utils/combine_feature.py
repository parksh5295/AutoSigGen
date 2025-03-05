# for Combine Processed Features
# Input data is 'feature_list'
import numpy as np


def combine_feature(feature_list):
    # e.g. for feature_list(input); [categorical_data, time_data, packet_length_data, count_data, binary_data]
    X = np.hstack(feature_list)
    return X