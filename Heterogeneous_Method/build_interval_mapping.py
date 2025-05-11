import pandas as pd


def build_interval_mapping_dataframe(group_mapping_info):
    print(f"[DEBUG] build_interval_mapping_dataframe: Received group_mapping_info for file_type (inferred): {group_mapping_info}") # DEBUG PRINT
    rows = []

    for feature, mapping in group_mapping_info.items():
        for interval, group in mapping.items():
            rows.append({
                'feature': feature,
                'interval=group': f"{interval}={group}"
            })

    if not rows: # Check if rows is empty after processing group_mapping_info
        print("[DEBUG] build_interval_mapping_dataframe: No rows were generated, likely because group_mapping_info was empty or led to no data. Returning empty DataFrame.")
        # Return an empty DataFrame with expected columns to avoid downstream errors if possible, or handle as needed.
        # For this specific function, an empty df_pivot might be acceptable if nothing to pivot.
        return pd.DataFrame(columns=['feature', 'interval=group', 'idx']).pivot(index='idx', columns='feature', values='interval=group')

    df = pd.DataFrame(rows)

    # Pivot so each column is a feature, and rows are interval=group strings
    # It's safer to check if 'feature' column actually exists if df might be unexpectedly formed
    if 'feature' not in df.columns:
        print(f"[DEBUG] build_interval_mapping_dataframe: 'feature' column missing from df. df columns: {df.columns}. df head: {df.head()}")
        # Decide how to handle this: raise error, return empty, etc.
        # For now, let's allow it to proceed to the groupby to see the original error if this check isn't the root cause,
        # but this state (rows populated but no 'feature' column) would be highly unusual given the append logic.
        pass # Or handle more gracefully, e.g., return empty pivot

    df['idx'] = df.groupby('feature').cumcount()
    df_pivot = df.pivot(index='idx', columns='feature', values='interval=group')

    return df_pivot # df_pivot = interval_mapping_info
