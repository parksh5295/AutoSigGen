import pandas as pd

# 1. Importing an existing CSV (with a space-separated string in the payload)
df1 = pd.read_csv("../../Dataset/load_dataset/NSL-KDD/train/train_payload_pre.csv")
df2 = pd.read_csv("../../Dataset/load_dataset/NSL-KDD/test/test_payload_pre.csv")

# 2. Divide payload by whitespace
payload_expanded1 = df1["payload"].str.split(" ", expand=True)
payload_expanded2 = df2["payload"].str.split(" ", expand=True)

payload_expanded1 = pd.concat([payload_expanded1, df1["binary_label"]], axis=1)
payload_expanded2 = pd.concat([payload_expanded2, df2["binary_label"]], axis=1)

# 3. 42 new column names
columns_42 = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# 4. Apply column names
payload_expanded1.columns = columns_42
payload_expanded2.columns = columns_42

# 5. Save as a new CSV
payload_expanded1.to_csv("../../Dataset/load_dataset/NSL-KDD/train/train_payload.csv", index=False)
payload_expanded2.to_csv("../../Dataset/load_dataset/NSL-KDD/test/test_payload.csv", index=False)
