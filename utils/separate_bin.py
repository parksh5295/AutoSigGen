# A collection of functions for splitting intervals

import pandas as pd


def interval_length_same(df, features):
    if not features:
        raise ValueError("Error: The `features` list is empty.")

    bin_df = pd.DataFrame()
    bins_count = 30
    group_mapping_info = {}
    labels = range(1, bins_count + 1)

    for i in range(len(features)):
        if features[i] not in df.columns:
            raise KeyError(f"Error: Column `{features[i]}` does not exist in `df`.")
        
        bin_df[features[i]] = pd.cut(df[features[i]], bins=bins_count, labels=labels, right=True)

        # Map indexes by bins -> to pass to mapping info later
        unique_intervals = bin_df[features[i]].dropna().unique()
        interval_to_group = {interval: i for i, interval in enumerate(sorted(unique_intervals))}
        group_mapping_info[features[i]] = interval_to_group

    return bin_df, group_mapping_info


def interval_length_Inverse_Count(df, features):
    if not features:
        raise ValueError("Error: The `features` list is empty.")

    bin_df = pd.DataFrame()
    bins_count = 30
    group_mapping_info = {}

    for i in range(len(features)):
        if features[i] not in df.columns:
            raise KeyError(f"Error: Column `{features[i]}` does not exist in `df`.")
        
        '''
        print(df[features[i]].describe())  # Check data distribution
        print(df[features[i]].value_counts())  # Check value frequency
        '''
        df[features[i]] = df[features[i]].rank(method="dense")  # Use rank() to sort the data first, then apply qcut
        bin_df[features[i]] = pd.qcut(df[features[i]], q=bins_count, labels=None, duplicates="drop")    # Group names automatically match the number of bins
        # If there are a lot of duplicate values, put them all in a group and treat them like categorical features.

        # Map indexes by bins -> to pass to mapping info later
        unique_intervals = bin_df[features[i]].dropna().unique()
        interval_to_group = {interval: i for i, interval in enumerate(sorted(unique_intervals))}
        group_mapping_info[features[i]] = interval_to_group

    return bin_df, group_mapping_info