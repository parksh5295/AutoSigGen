import pandas as pd
import numpy as np


# Estimate dtype while memory-efficiently sampling a CSV
def infer_dtypes_safely(file_type, csv_path, max_rows=1000, chunk_size=100):
    '''
    if file_type in ['DARPA98', 'DARPA']:
        force_str_columns = {'Date', 'StartTime', 'Duration'}
    else:
        force_str_columns = None
    '''
    
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

                '''
                # If the column should be forced to str, add "str" to its dtype set
                if force_str_columns is not None and col in force_str_columns:
                    inferred_dtypes[col].add("str")
                    continue
                '''

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

    if inferred_dtypes is None: # Handle case where CSV might be empty or smaller than max_rows
        print("[WARN] infer_dtypes_safely: Could not infer dtypes, possibly empty or very small CSV. Returning empty dtype_map.")
        return {}

    print("[DEBUG] inferred_dtypes type:", type(inferred_dtypes))
    print("[DEBUG] inferred_dtypes sample:", str(inferred_dtypes)[:500])

    # dtype estimation rules
    dtype_map = {}
    for col, types in inferred_dtypes.items():
        if not types: # If a column had all nulls in the sample
            dtype_map[col] = 'object' # Default to object, or handle as per desired logic
            print(f"[WARN] Column '{col}' had all nulls in sample, defaulting to object type.")
        elif types <= {"int"}:
            dtype_map[col] = 'int32'
        elif types <= {"int", "float"}:
            dtype_map[col] = 'float32'
        elif types <= {"str"}:
            dtype_map[col] = 'object'
        else:
            print(f"[WARN] Unknown type set {types} for column '{col}', using object")
            dtype_map[col] = 'object'
    return dtype_map


def _post_process_specific_datasets(df, file_type):
    """
    Applies specific post-loading transformations based on file_type.
    """
    if file_type == 'IoTID20':
        if 'Timestamp' in df.columns:
            try:
                print(f"[INFO] dtype_optimize: Post-processing 'Timestamp' for IoTID20.")
                # Assuming format 'DD/MM/YYYY HH:MM:SS AM/PM' (e.g., 11/07/2019 01:29:09 AM)
                # If format is 'MM/DD/YYYY...', change to '%m/%d/%Y %I:%M:%S %p'
                df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
                
                valid_timestamps = df['Timestamp_dt'].notna()
                # Use .loc to assign and avoid SettingWithCopyWarning
                df.loc[valid_timestamps, 'Timestamp'] = (df.loc[valid_timestamps, 'Timestamp_dt'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                df.loc[~valid_timestamps, 'Timestamp'] = np.nan # Assign NaN to unparseable timestamps

                df.drop(columns=['Timestamp_dt'], inplace=True, errors='ignore')
                # Ensure the final 'Timestamp' column is numeric, coercing any remaining issues to NaN
                df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
                print(f"[INFO] dtype_optimize: 'Timestamp' column post-processed for IoTID20. NaN count: {df['Timestamp'].isnull().sum()}")
            except Exception as e:
                print(f"[ERROR] dtype_optimize: Failed to post-process 'Timestamp' for IoTID20. Error: {e}")
        else:
            print("[WARN] dtype_optimize: 'Timestamp' column not found in IoTID20 data during post-processing.")
    
    # Add other dataset-specific post-processing here if needed
    # elif file_type == 'OtherDataset':
    #     ...
        
    return df


# Efficiently load a full CSV into a DataFrame after auto-estimating dtype
def load_csv_safely(file_type, csv_path, max_rows_for_inference=1000):
    print("[INFO] Estimating: Sampling for dtype inference...")
    dtype_map = infer_dtypes_safely(file_type, csv_path, max_rows=max_rows_for_inference)
    print("dtype map: ", dtype_map)
    print("[INFO] Dtype inference complete. Loading full CSV...")
    df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=False)
    print("[INFO] Finished loading the DataFrame:", df.shape)

    # Apply dataset-specific post-processing
    df = _post_process_specific_datasets(df, file_type)

    return df
