# Functions to proactively remove rare items
def remove_rare_columns(df, min_support_ratio, file_type=None, min_distinct_frequent_values=2):
    '''
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD is applied with a lower threshold (only 20% of the original min_support_ratio)
        threshold = int(len(df) * min_support_ratio * 0.2)
    else:
    '''
    threshold = int(len(df) * min_support_ratio)
    
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD modified logic
        original_cols = df.columns
        valid_cols = []
        for col in original_cols:
            value_counts = df[col].value_counts()
            # Count the number of unique values above a threshold
            count_distinct_frequent = sum(1 for count in value_counts if count >= threshold)

            # Keep column only if there are at least min_distinct_frequent_values unique values above threshold
            if count_distinct_frequent >= min_distinct_frequent_values:
                valid_cols.append(col)

        print(f"Original columns: {len(original_cols)}")
        print(f"Threshold value (for individual value frequency): {threshold}")
        print(f"Required distinct frequent values: {min_distinct_frequent_values}")
        print(f"Remaining columns after filtering: {len(valid_cols)}")

        if len(valid_cols) == 0:
            print("Warning: All columns were filtered out! Using original columns instead.")
            return df

        # Safeguards: If too many columns are removed (e.g., fewer than 5), consider keeping the originals
        if len(valid_cols) < 5 and len(original_cols) >= 5 :
             print(f"Warning: Too few columns remaining ({len(valid_cols)}). Falling back to original columns to avoid issues.")
             return df

        return df[valid_cols]
    else:
        # Keep the original logic
        threshold = int(len(df) * min_support_ratio)
        valid_cols = df.columns[df.sum() > threshold]
        return df[valid_cols]

