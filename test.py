import pandas as pd


# with open('D:\AutoSigGen_withData\Dataset\load_dataset\CICIDS2017\MachineLearningCSV\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'r') as f:
csv_path = 'D:\AutoSigGen_withData\Dataset\load_dataset\DARPA98\\train\\1w_friday\\1w_friday_pre.csv'
column_names = pd.read_csv(csv_path, nrows=6)

print("Column Names:")
print(column_names)