# Agendas for importing partial or full files in a dataset
# Output is data

import pandas as pd
import random
from Dataset_Choose_Rule.CICIDS2017_csv_selector import select_csv_file
from Dataset_Choose_Rule.dtype_optimize import infer_dtypes_safely, _post_process_specific_datasets


def file_path_line_nonnumber(file_type, file_number=1): # file_number is not used, but insert to prevent errors from occurring
    if file_type == 'MiraiBotnet':
        file_path = "../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv"
    elif file_type in ['ARP', 'MitM', 'Kitsune']:
        file_path = "../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
    elif file_type in ['CICIDS2017', 'CICIDS']:
        file_path, file_number =  select_csv_file()
    elif file_type == 'netML' :
        file_path = "../Dataset/load_dataset/netML/netML_dataset.csv"
    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        file_path = "../Dataset/load_dataset/NSL-KDD/NSL-KDD_dataset.csv"
    elif file_type in ['DARPA', 'DARPA98']:
        file_path = "../Dataset/load_dataset/DARPA98/train/DARPA98.csv"
    elif file_type in ['CICModbus23', 'CICModbus']:
        file_path = "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    elif file_type in ['IoTID20', 'IoTID']:
        file_path = "../Dataset/load_dataset/IoTID20/IoTID20.csv"
    else:
        print("No file information yet, please double-check the file type or provide new data!")
        file_path_line_nonnumber(file_type)
    return file_path, file_number

def file_path_line_withnumber(file_type, file_number=1):
    return # file_path

# After selecting the file path
# Functions for getting only part of a file as data
def file_cut(file_type, file_path, cut_type='random'):
    inferred_dtypes = infer_dtypes_safely(file_type, file_path)

    df = None  # Initialize df to avoid UnboundLocalError

    if cut_type == 'random':
        # Get the total number of rows (excluding headers)
        total_rows = sum(1 for _ in open(file_path)) - 1  # excluding headers

        # Select row numbers to randomly sample
        # num_rows_to_sample = int(input("Enter the desired number of rows of data: "))
        num_rows_to_sample = 3000
        sampled_rows = sorted(random.sample(range(1, total_rows + 1), num_rows_to_sample))

        # Read only selected rows (but keep headers)
        df = pd.read_csv(
            file_path,
            dtype=inferred_dtypes,
            skiprows=lambda x: x > 0 and x not in sampled_rows
        )
        df = _post_process_specific_datasets(df, file_type)

    elif cut_type in ['in order', 'In order', 'In Order']:    # from n~m row
        n = int(input("Enter the row number to start with: "))  # Row number to start with (1-based index, i.e., first data is 1)
        m = int(input("Enter the row number to end with: "))  # Row number to end

        df = pd.read_csv(
            file_path,
            dtype=inferred_dtypes,
            skiprows=lambda x: x > 0 and x < n,
            nrows=m - n + 1
        )
        df = _post_process_specific_datasets(df, file_type)

    elif cut_type in ['all', 'All']:
        df = pd.read_csv(file_path, dtype=inferred_dtypes)
        df = _post_process_specific_datasets(df, file_type)

    return df   # return data