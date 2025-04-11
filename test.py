import pandas as pd


chunk_size = 10
for chunk in pd.read_csv('../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv', chunksize=chunk_size):
    print(chunk.head(6))