# Convert date, time to integer

import pandas as pd
import numpy as np


def correct_invalid_date(date_str, is_two_digit_year=False):
    """Fix invalid dates to the nearest valid date"""
    try:
        month, day, year = map(int, date_str.split('/'))
        # Last date of each month
        month_last_day = {
            1: 31, 2: 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        # Date calibration
        if day > month_last_day[month]:
            day = month_last_day[month]
            
        if is_two_digit_year:
            return f"{month:02d}/{day:02d}/{year:02d}"
        else:
            return f"{month:02d}/{day:02d}/{year:04d}"
    except:
        return date_str

def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        data['Date'] = data['Date'].astype(str).str.strip()
        data['StartTime'] = data['StartTime'].astype(str).str.strip()
        data['Duration'] = data['Duration'].astype(str).str.strip()

        regex2 = r'^\d{2}/\d{2}/\d{2}$'
        regex4 = r'^\d{2}/\d{2}/\d{4}$'
        
        mask2 = data['Date'].str.match(regex2)
        mask4 = data['Date'].str.match(regex4)

        try:
            data.loc[mask2, 'Date'] = pd.to_datetime(data.loc[mask2, 'Date'], format="%m/%d/%y")
        except ValueError:
            problematic_dates = data.loc[mask2, 'Date']
            corrected_dates = problematic_dates.apply(lambda x: correct_invalid_date(x, True))
            data.loc[mask2, 'Date'] = pd.to_datetime(corrected_dates, format="%m/%d/%y")

        try:
            data.loc[mask4, 'Date'] = pd.to_datetime(data.loc[mask4, 'Date'], format="%m/%d/%Y")
        except ValueError:
            problematic_dates = data.loc[mask4, 'Date']
            corrected_dates = problematic_dates.apply(lambda x: correct_invalid_date(x, False))
            data.loc[mask4, 'Date'] = pd.to_datetime(corrected_dates, format="%m/%d/%Y")

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