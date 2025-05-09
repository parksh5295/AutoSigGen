import argparse
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
import pandas as pd

def check_kitsune_label_distribution(file_number_to_test=1):
    """Loads a specific Kitsune dataset file and prints its label distribution."""
    file_type = "Kitsune"
    print(f"\nTesting Kitsune dataset, file number: {file_number_to_test}")

    try:
        file_path, _ = file_path_line_nonnumber(file_type, file_number_to_test) # Ignore the second return value with _.
        # cut_type is assumed to be 'all' for Kitsune (see Data_Labeling.py logic)
        # assumes file_cut function only works with file path and file type.
        # You may need to adapt to the actual arguments of file_cut.
        data = file_cut(file_type, file_path, 'all')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n--- Kitsune Dataset Label Distribution Test ---")
    if 'Label' in data.columns:
        # The labels may not be numeric, so you may want to consider checking or converting them before value_counts().
        # Here we assume that the Label column contains 0s and 1s in numeric form.
        if pd.api.types.is_numeric_dtype(data['Label']):
            label_counts = data['Label'].value_counts().sort_index()
            print("Label counts in the loaded Kitsune dataset:")
            print(label_counts)
            if 0 not in label_counts:
                print("WARNING: Label 0 (normal) is not present in the loaded data.")
            else:
                print(f"Count of Label 0 (normal): {label_counts.get(0, 0)}")
            if 1 not in label_counts:
                print("WARNING: Label 1 (anomalous) is not present in the loaded data.")
            else:
                print(f"Count of Label 1 (anomalous): {label_counts.get(1, 0)}")
        else:
            print("ERROR: 'Label' column is not numeric. Cannot perform count for 0 and 1 directly.")
            print("Actual label distribution:")
            print(data['Label'].value_counts().sort_index())

    else:
        print("ERROR: 'Label' column not found in the loaded Kitsune dataset.")
    print("--- End of Kitsune Dataset Label Distribution Test ---\n")

if __name__ == '__main__':
    # Run the test against a specific file number.
    # To test against multiple files, you can use a loop, for example.
    # Or you can use argparse to get the file number from the command line.
    parser = argparse.ArgumentParser(description='Test Kitsune Label Distribution')
    parser.add_argument('--file_number', type=int, default=1, help='File number of the Kitsune dataset to test')
    args = parser.parse_args()

    check_kitsune_label_distribution(args.file_number) 