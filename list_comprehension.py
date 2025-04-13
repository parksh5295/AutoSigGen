import csv
import glob


# Get paths to list files (e.g., *.list)
list_files = glob.glob('D:\\AutoSigGen_withData\\Dataset\\load_dataset\\DARPA98\\train\*\*.list')

output_csv = 'D:\\AutoSigGen_withData\\Dataset\\load_dataset\\DARPA98\\train\\DARPA98.csv'

columns = ['ID', 'Date', 'StartTime', 'Duration', 'Protocol', 'SrcPort', 'DstPort', 'SrcIP', 'DstIP', 'Flag', 'Class']

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)

    for file in list_files:
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip comments or blank lines
                    continue
                parts = line.split()
                if len(parts) >= 11:
                    writer.writerow(parts[:11])
