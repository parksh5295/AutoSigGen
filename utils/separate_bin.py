# A collection of functions for splitting intervals

import pandas as pd


def interval_length_same(df, features):
    bin_df = pd.DataFrame()
    bins_count = 30
    labels = range(1, bins_count + 1)
    for i in range(len(features)):
        bin_df[features[i]] = pd.cut(df[features[i]], bins=bins_count, labels=labels, right=True)
    return bin_df


def interval_length_Inverse_Count(df, features):
    bin_df = pd.DataFrame()
    bins_count = 30
    labels = range(1, bins_count + 1)
    for i in range(len(features)):
        bin_df[features[i]] = pd.qcut(df[features[i]], q=bins_count, labels=labels)
    return bin_df