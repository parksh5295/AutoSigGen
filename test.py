import csv

with open('D:\AutoSigGen_withData\Dataset\load_dataset\DARPA98\\train\DARPA98.csv', 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader)
print("Row count: ", row_count)
