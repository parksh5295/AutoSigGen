# Convert date, time to integer

import pandas as pd


def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        # date transfer
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data['Date_scalar'] = data['Date'].astype('int64')

        # StartTime transfer
        data['StartTime_scalar'] = data['StartTime'].apply(lambda x: sum(
            int(t) * 60 ** i for i, t in enumerate(reversed(x.split(":")))
        ))

        # Duration transfer
        data['Duration_scalar'] = data['Duration'].apply(lambda x: sum(
            int(t) * 60 ** i for i, t in enumerate(reversed(x.split(":")))
        ))

        return data
    
    else:
        pass