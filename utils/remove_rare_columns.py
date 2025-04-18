# Functions to proactively remove rare items
def remove_rare_columns(df, min_support_ratio, file_type=None):
    threshold = int(len(df) * min_support_ratio)
    
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD modified logic
        valid_cols = []
        for col in df.columns:
            value_counts = df[col].value_counts()
            if any(count >= threshold for count in value_counts):
                valid_cols.append(col)
        
        print(f"Original columns: {len(df.columns)}")
        print(f"Remaining columns after filtering: {len(valid_cols)}")
        
        return df[valid_cols]
    else:
        # Keep the original logic
        valid_cols = df.columns[df.sum() > threshold]
        return df[valid_cols]

