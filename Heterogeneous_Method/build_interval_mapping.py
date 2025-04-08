import pandas as pd


def build_interval_mapping_dataframe(group_mapping_info):
    rows = []

    for feature, mapping in group_mapping_info.items():
        for interval, group in mapping.items():
            rows.append({
                'feature': feature,
                'interval=group': f"{interval}={group}"
            })

    df = pd.DataFrame(rows)

    # Pivot so each column is a feature, and rows are interval=group strings
    df['idx'] = df.groupby('feature').cumcount()
    df_pivot = df.pivot(index='idx', columns='feature', values='interval=group')

    return df_pivot # df_pivot = interval_mapping_info
