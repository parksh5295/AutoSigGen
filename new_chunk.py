# python file to generate chunks for feature checking

import pandas as pd


chunk_size = 10
# output_file = '../Dataset/chunk_dataset/MiraiBotnet_chunks.csv'
# output_file = '../Dataset/chunk_dataset/Kitsune_chunks.csv'
# output_file = '../Dataset/chunk_dataset/netML_dataset.csv'
output_file = '../Dataset/chunk_dataset/CICIDS2017_dataset.csv'

# Overwrite only 6 lines in each batch
# for i, chunk in enumerate(pd.read_csv('../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv', chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv", chunksize=chunk_size)):
# for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/netML/netML_dataset.csv", chunksize=chunk_size)):
for i, chunk in enumerate(pd.read_csv("../Dataset/load_dataset/CICIDS2017/MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", chunksize=chunk_size)):
    first_6_rows = chunk.head(8)
    
    first_6_rows.to_csv(output_file, mode='w', header=True, index=False)    # Save as overwrite on redo