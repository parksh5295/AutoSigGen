# Convert date, time to integer

import pandas as pd


def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        data['Date'] = data['Date'].astype(str)
        data['StartTime_scalar'] = data['StartTime_scalar'].astype(str)
        data['Duration_scalar'] = data['Duration_scalar'].astype(str)

        # date transfer
        try:
            # First try with a 2-digit year
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
        except ValueError:
            try:
                # If that doesn't work, try a 4-digit year
                data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            except Exception as e:
                print(f"[ERROR] Failed to parse 'Date': {e}")

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