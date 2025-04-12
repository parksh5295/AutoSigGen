# with open('D:\AutoSigGen_withData\Dataset\load_dataset\CICIDS2017\MachineLearningCSV\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'r') as f:
with open('D:\AutoSigGen_withData\Dataset\load_dataset\NSL-KDD\NSL-KDD_dataset.csv', 'r') as f:
    row_count = sum(1 for line in f)
    print("count row: ", row_count)