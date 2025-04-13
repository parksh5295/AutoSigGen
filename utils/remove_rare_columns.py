

# Functions to proactively remove rare items
def remove_rare_columns(df, min_support_ratio):
    threshold = int(len(df) * min_support_ratio)
    valid_cols = df.columns[df.sum() > threshold]
    return df[valid_cols]

