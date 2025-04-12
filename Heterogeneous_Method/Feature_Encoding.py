# Separating features into attributes and encoding each
# Output data is 'feature_list'
# Encoding and Normalization

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd


def Heterogeneous_Feature_named_featrues(file_type):
    if file_type == 'MiraiBotnet':
        categorical_features = [
            'flow_protocol'
            ]
        time_features = [
            'flow_iat_max', 'flow_iat_min', 'flow_iat_mean', 'flow_iat_total', 'flow_iat_std',
            'forward_iat_max', 'forward_iat_min', 'forward_iat_mean', 'forward_iat_total', 'forward_iat_std',
            'backward_iat_max', 'backward_iat_min', 'backward_iat_mean', 'backward_iat_total', 'backward_iat_std'
            ]
        packet_length_features = [
            'total_bhlen', 'total_fhlen',
            'forward_packet_length_mean', 'forward_packet_length_min', 'forward_packet_length_max', 'forward_packet_length_std',
            'backward_packet_length_mean', 'backward_packet_length_min', 'backward_packet_length_max', 'backward_packet_length_std'
        ]
        count_features = [
            'fpkts_per_second', 'bpkts_per_second', 'total_forward_packets', 'total_backward_packets',
            'total_length_of_forward_packets', 'total_length_of_backward_packets', 'flow_packets_per_second'
        ]
        binary_features = [
            'flow_psh', 'flow_syn', 'flow_urg', 'flow_fin', 'flow_ece', 'flow_ack', 'flow_rst', 'flow_cwr'
        ]

    elif file_type in ['MitM', 'Kitsune']:
        categorical_features = []
        time_features = [
            'SrcMAC_IP_w_100ms', 'SrcMAC_IP_mu_100ms', 'SrcMAC_IP_sigma_100ms', 'SrcMAC_IP_max_100ms', 'SrcMAC_IP_min_100ms',
            'SrcIP_w_100ms', 'SrcIP_mu_100ms', 'SrcIP_sigma_100ms', 'SrcIP_max_100ms', 'SrcIP_min_100ms',
            'Channel_w_100ms', 'Channel_mu_100ms', 'Channel_sigma_100ms', 'Channel_max_100ms', 'Channel_min_100ms'
        ]
        packet_length_features = [
            'Socket_w_100ms', 'Socket_mu_100ms', 'Socket_sigma_100ms', 'Socket_max_100ms', 'Socket_min_100ms'
        ]
        count_features = [
            'Jitter_mu_100ms', 'Jitter_sigma_100ms', 'Jitter_max_100ms'
        ]
        binary_features = []

    elif file_type in ['CICIDS2017', 'CICIDS']:
        categorical_features = ['Destination Port']
        time_features = [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        packet_length_features = [
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
        count_features = [
            'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Fwd Header Length', 'Bwd Header Length',
            'Down/Up Ratio', 'Average Packet Size',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward'
        ]
        binary_features = [
            'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ]

    elif file_type == 'netML':
        categorical_features = ['Protocol']
        time_features = [
            'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min'
        ]
        packet_length_features = [
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std'
        ]
        count_features = [
            'Total Fwd Packets', 'Total Backward Packets', 'Flow Packets/s',
            'Fwd Packets/s', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Subflow Bwd Packets'
        ]
        binary_features = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count'
        ]

    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        categorical_features = ['protocol_type', 'service', 'flag']
        time_features = []
        packet_length_features = [
            'src_bytes', 'dst_bytes'
        ]
        count_features = [
            'duration', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        binary_features = [
            'land', 'logged_in', 'is_host_login', 'is_guest_login'
        ]
    
    return {
            'categorical_features': categorical_features,
            'time_features': time_features,
            'packet_length_features': packet_length_features,
            'count_features': count_features,
            'binary_features': binary_features
        }   


def Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data):
    encoder = LabelEncoder()

    if not categorical_features:
        categorical_data = np.empty((len(data), 0))
    else:
        categorical_data = pd.DataFrame()
        categorical_mapping_info = {}
        for col in categorical_features:
            categorical_data[col] = encoder.fit_transform(data[col]) + 1  # Mapping to values starting at 1
            categorical_mapping_info[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_) + 1))

        # Organize mapping information in a value=number format
        max_len = max(len(mapping) for mapping in categorical_mapping_info.values())
        formatted_columns = {}
        for feature, mapping in categorical_mapping_info.items():
            items = [f"{k}={v}" for k, v in mapping.items()]
            items += [""] * (max_len - len(items))  # Align length
            formatted_columns[feature] = items

        categorical_mapping_df = pd.DataFrame(formatted_columns)

    if not time_features:
        time_data = np.empty((len(data), 0))
    else:
        time_data = data[time_features]

    if not packet_length_features:
        packet_length_data = np.empty((len(data), 0))
    else:
        packet_length_data = data[packet_length_features]

    if not count_features:
        packet_count_data = np.empty((len(data), 0))
    else:
        packet_count_data = data[count_features]

    if not binary_features:
        flow_flag_data = np.empty((len(data), 0))
    else:
        flow_flag_data = data[binary_features]
        binary_mapping_info = {}
        for col in binary_features:
            binary_mapping_info[col] = {0: 0, 1: 1}

        max_len = max(len(mapping) for mapping in binary_mapping_info.values())
        formatted_binary = {}
        for feature, mapping in binary_mapping_info.items():
            items = [f"{k}={v}" for k, v in mapping.items()]
            items += [""] * (max_len - len(items))
            formatted_binary[feature] = items

        binary_mapping_df = pd.DataFrame(formatted_binary)

    # Combine all processed data into a list
    data_list = [categorical_data, time_data, packet_length_data, packet_count_data, flow_flag_data]
    category_mapping = {
        'categorical': categorical_mapping_df,
        'binary': binary_mapping_df
    }

    return data_list, category_mapping


def Heterogeneous_Feature_named_combine_standard(categorical_features, time_features, packet_length_features, count_features, binary_features, data):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()

    if not categorical_features:
        categorical_data = np.empty((len(data), 0))
    else:
        categorical_data = encoder.fit_transform(data[categorical_features])    # Returns as a numpy.ndarray type
        categorical_data = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_features))  # Switching to pandas dataframe

    if not time_features:
        time_data = np.empty((len(data), 0))
    else:
        time_data = scaler.fit_transform(data[time_features])
        time_data = pd.DataFrame(time_data, columns=time_features)  # Using original feature names

    if not packet_length_features:
        packet_length_data = np.empty((len(data), 0))
    else:
        packet_length_data = scaler.fit_transform(data[packet_length_features])
        packet_length_data = pd.DataFrame(packet_length_data, columns=packet_length_features)  # Using original feature names

    if not count_features:
        packet_count_data = np.empty((len(data), 0))
    else:
        packet_count_data = scaler.fit_transform(data[count_features])
        packet_count_data = pd.DataFrame(packet_count_data, columns=count_features)  # Using original feature names

    if not binary_features:
        flow_flag_data = np.empty((len(data), 0))
    else:
        flow_flag_data = pd.DataFrame(data[binary_features].astype(int), columns=binary_features)

    # Combine all processed data into a list
    data_list = [categorical_data, time_data, packet_length_data, packet_count_data, flow_flag_data]

    return data_list