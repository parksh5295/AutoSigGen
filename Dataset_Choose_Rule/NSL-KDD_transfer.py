import pandas as pd
from scipy.io import arff

# 1. Reading ARFF Files
data, meta = arff.loadarff('../../Dataset/load_dataset/NSL-KDD/KDDTest+.arff')

# 2. Converting byte data to strings (especially where nominal properties may be in the form of b'normal')
df = pd.DataFrame(data)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# 3. Specify column names (must be provided in the same order as ARFF)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]
df.columns = columns

# 4. Save to CSV
df.to_csv('../../Dataset/load_dataset/NSL-KDD/NSL-KDD_dataset.csv', index=False)
