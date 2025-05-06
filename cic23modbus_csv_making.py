import os
import glob
import pandas as pd

def load_and_merge_cicmodbus_attacks(base_dir):
    all_data = []
    attack_categories = ['compromised-ied', 'compromised-scada', 'external']
    
    for category in attack_categories:
        path = os.path.join(base_dir, category, 'attack_logs', '**', '*.csv')
        csv_files = glob.glob(path, recursive=True)

        print(f"[{category}] Found {len(csv_files)} files")

        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['attack_type'] = category
                all_data.append(df)
            except Exception as e:
                print(f"Failed to read {file}: {e}")

    if not all_data:
        raise ValueError("No CSV files loaded. Check the base directory path and structure.")
    
    full_df = pd.concat(all_data, ignore_index=True)
    
    if 'Timestamp' not in full_df.columns:
        raise KeyError("Missing 'Timestamp' column in CSVs. Please check column names.")

    # Attempt to parse Timestamp column, count errors, and drop column if parsing fails
    timestamps = pd.to_datetime(full_df['Timestamp'], errors='coerce')
    n_errors = timestamps.isna().sum()
    if n_errors > 0:
        print(f"Dropping 'Timestamp' column due to {n_errors} parse errors.")
        full_df.drop(columns=['Timestamp'], inplace=True)
    else:
        full_df['Timestamp'] = timestamps
        full_df.sort_values(by='Timestamp', inplace=True)

    # Ensure TargetIP and TransactionID columns exist; fill with dummy values for external logs
    # For logs missing these columns, assign defaults
    if 'TargetIP' not in full_df.columns:
        full_df['TargetIP'] = 'unknown'
    else:
        full_df['TargetIP'].fillna('unknown', inplace=True)
    if 'TransactionID' not in full_df.columns:
        full_df['TransactionID'] = -1
    else:
        full_df['TransactionID'].fillna(-1, inplace=True)
    
    return full_df


if __name__ == "__main__":
    base_dir = r"..\Dataset\load_dataset\CICModbus23\attack"  # e.g., r"D:\AutoSigGen_withData\Dataset\load_dataset\CICModbus23\attack"
    output_csv = r"..\Dataset\load_dataset\CICModbus23\CICModbus23_total.csv"  # e.g., "merged_modbus.csv"

    df = load_and_merge_cicmodbus_attacks(base_dir)
    # Save merged DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved merged DataFrame to {output_csv}, shape: {df.shape}")

    
