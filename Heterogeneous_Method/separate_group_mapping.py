# Input embedded_data (after separate_bin)

import pandas as pd
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe


def map_intervals_to_groups(df, category_mapping, data_list, regul='N'):
    mapped_df = pd.DataFrame()  # Save the data converted to group numbers
    mapping_info = {}   # Save per-feature mapping information

    interval_columns = list(category_mapping['interval'].keys())
    print("category_mapping[0]: ", category_mapping['categorical'])
    print("category_mapping[1]: ", category_mapping['interval'])
    print("category_mapping[2]: ", category_mapping['binary'])
    interval_df = df[interval_columns]  # Organize only the conditions that want to map

    interval_mapping = category_mapping.get('interval', {})  # interval information is taken from here

    for col in interval_df.columns:
        if col not in interval_mapping.columns:
            raise KeyError(f"Interval mapping for column `{col}` is missing.")
        
        # Extracting columns from a DataFrame
        series = interval_mapping[col].dropna()

        # interval=group format should be separated and made into a dict
        # ex: "(0.999, 17.0]=0" → key="(0.999, 17.0]", value=0
        interval_to_group = {}
        for s in series:
            try:
                interval_str, group_num = s.split('=')
                interval_to_group[interval_str.strip()] = int(group_num.strip())
            except ValueError:
                print(f"Invalid format in mapping: {s}")

        mapping_info[col] = interval_to_group

        mapped_df[col] = interval_df[col].astype(str).map(interval_to_group)    # The value of df[col] must actually be a string for the mapping to work

    mapped_df = pd.concat([data_list[0], mapped_df, data_list[len(data_list)-1]], axis=1)

    mapped_info_df = build_interval_mapping_dataframe(mapping_info)

    if regul in ["N", "n"]:
        mapped_info_df = pd.concat([
            category_mapping.get('categorical', pd.DataFrame()),
            mapped_info_df,
            category_mapping.get('binary', pd.DataFrame())
        ], axis=1, ignore_index=False)

    return mapped_df, mapped_info_df
