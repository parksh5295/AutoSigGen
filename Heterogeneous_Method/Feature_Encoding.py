# Separating features into attributes and encoding each
# Output data is 'feature_list'
# Encoding and Normalization

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np


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

    elif file_type == 'MitM':
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

    elif file_type == 'CICIDS2017':
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
    
    return {
            'categorical_features': categorical_features,
            'time_features': time_features,
            'packet_length_features': packet_length_features,
            'count_features': count_features,
            'binary_features': binary_features
        }   


def Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()

    if not categorical_features:
        categorical_data = np.empty((len(data), 0))
    else:
        categorical_data = encoder.fit_transform(data[categorical_features])

    if not time_features:
        time_data = np.empty((len(data), 0))
    else:
        time_data = scaler.fit_transform(data[time_features])

    if not packet_length_features:
        packet_length_data = np.empty((len(data), 0))
    else:
        packet_length_data = scaler.fit_transform(data[packet_length_features])

    if not count_features:
        packet_count_data = np.empty((len(data), 0))
    else:
        packet_count_data = scaler.fit_transform(data[count_features])

    if not binary_features:
        flow_flag_data = np.empty((len(data), 0))
    else:
        flow_flag_data = data[binary_features].astype(int)

    data_list = [categorical_data, time_data, packet_length_data, packet_count_data, flow_flag_data]

    return data_list