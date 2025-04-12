import pandas as pd
import numpy as np


# Estimate dtype while memory-efficiently sampling a CSV
def infer_dtypes_safely(csv_path, max_rows=1000, chunk_size=100):
    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    inferred_dtypes = None
    row_count = 0

    for chunk in chunk_iter:
        if inferred_dtypes is None:
            column_names = list(chunk.columns)
            inferred_dtypes = {col: set() for col in column_names}

        for _, row in chunk.iterrows():
            for col in column_names:
                val = row[col]
                if pd.isnull(val):
                    continue
                if isinstance(val, int):
                    inferred_dtypes[col].add("int")
                elif isinstance(val, float):
                    inferred_dtypes[col].add("float")
                elif isinstance(val, str):
                    inferred_dtypes[col].add("str")
                else:
                    inferred_dtypes[col].add("other")
            row_count += 1
            if row_count >= max_rows:
                break
        if row_count >= max_rows:
            break

    print("[DEBUG] inferred_dtypes type:", type(inferred_dtypes))
    print("[DEBUG] inferred_dtypes sample:", str(inferred_dtypes)[:500])

    # dtype estimation rules
    dtype_map = {}
    for col, types in inferred_dtypes.items():
        if types <= {int}:
            dtype_map[col] = 'int32'  # Use np.int32 instead of 'int32'
        elif types <= {int, float}:
            dtype_map[col] = 'float32'  # Use np.float32 instead of 'float32'
        elif types <= {str}:
            dtype_map[col] = 'category'
        else:
            dtype_map[col] = 'object'

    return dtype_map


# Efficiently load a full CSV into a DataFrame after auto-estimating dtype
def load_csv_safely(csv_path, max_rows_for_inference=1000):
    print("[INFO] Estimating: Sampling for dtype inference...")
    inferred_dtypes = infer_dtypes_safely(csv_path, max_rows=max_rows_for_inference)
    print("[INFO] Dtype inference complete. Loading full CSV...")
    df = pd.read_csv(csv_path, dtype=inferred_dtypes, low_memory=False)
    print("[INFO] Finished loading the DataFrame:", df.shape)
    return df
