# Convert date, time to integer

import pandas as pd


def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        data['Date'] = data['Date'].astype(str).str.strip()
        data['StartTime'] = data['StartTime'].astype(str).str.strip()
        data['Duration'] = data['Duration'].astype(str).str.strip()

        regex2 = r'^\d{2}/\d{2}/\d{2}$'
        regex4 = r'^\d{2}/\d{2}/\d{4}$'
        
        mask2 = data['Date'].str.match(regex2)
        mask4 = data['Date'].str.match(regex4)

        data.loc[mask2, 'Date'] = pd.to_datetime(data.loc[mask2, 'Date'],  format="%m/%d/%y")
        data.loc[mask4, 'Date'] = pd.to_datetime(data.loc[mask4, 'Date'],  format="%m/%d/%Y")
        print('[DEBUG] datetime conversion complete')

        data['Date_scalar'] = data['Date']

        # # date transfer
        # try:
        #     # First try with a 2-digit year
        #     data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        # except Exception as e1:
        #     print(f"[DEBUG] First format failed. Error type: {type(e1).__name__}")
        #     # print(f"[DEBUG] at {data}")
        #     print(f"[DEBUG] Error message: {e1}")
        #     try:
        #         # Then try with a 4-digit year
        #         data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')
        #     except Exception as e2:
        #         print(f"[DEBUG] Second format failed. Error type: {type(e2).__name__}")
        #         # print(f"[DEBUG] at {data}")
        #         print(f"[DEBUG] Error message: {e2}")

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