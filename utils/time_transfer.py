# Convert date, time to integer

import pandas as pd


def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        # date transfer
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data['Date_scalar'] = data['Date'].astype('int64') // 10**9

        # Time string to seconds
        def hms_to_seconds(hms_str):
            parts = [int(x) for x in hms_str.strip().split(":")]
            while len(parts) < 3:
                parts = [0] + parts
            h, m, s = parts
            return h * 3600 + m * 60 + s

        data['StartTime_scalar'] = data['StartTime'].apply(hms_to_seconds)
        data['Duration_scalar'] = data['Duration'].apply(hms_to_seconds)

        return data
    
    else:
        return data