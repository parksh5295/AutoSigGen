import pandas as pd

df = pd.read_csv("D:\\AutoSigGen_withData\\Dataset\\load_dataset\\CICModbus23\\CICModbus23_total.csv")
unique_attacks = df['Attack'].dropna().unique()

for attack in unique_attacks:
    print(repr(attack))