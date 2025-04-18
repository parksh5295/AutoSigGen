# Functions to proactively remove rare items
def remove_rare_columns(df, min_support_ratio, file_type=None):
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD is applied with a lower threshold (only 20% of the original min_support_ratio)
        threshold = int(len(df) * min_support_ratio * 0.2)
    else:
        threshold = int(len(df) * min_support_ratio)
    
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD modified logic
        valid_cols = []
        for col in df.columns:
            value_counts = df[col].value_counts()
            if any(count >= threshold for count in value_counts):
                valid_cols.append(col)
        
        print(f"Original columns: {len(df.columns)}")
        print(f"Threshold value: {threshold}")
        print(f"Remaining columns after filtering: {len(valid_cols)}")
        if len(valid_cols) == 0:
            print("Warning: All columns were filtered out! Using original columns instead.")
            return df  # If all columns are filtered out, return the original data
        
        return df[valid_cols]
    else:
        # Keep the original logic
        valid_cols = df.columns[df.sum() > threshold]
        return df[valid_cols]

