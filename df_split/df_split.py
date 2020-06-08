
import random
import pandas as pd
import numpy as np

def random_split_msk(df, train_frac = 0.8):
    k = int(df.shape[0] * train_frac)
    idxs = random.sample(list(range(df.shape[0])), k)
    train_msk = np.array([i in list(range(df.shape[0])) for i in idxs])
    test_msk = ~train_msk

    return  train_msk, test_msk

def date_split_msk(date_data, test_days, start_from = None):
    if not start_from:
        start_from = date_data.max()
    else:
        start_from = pd.to_datetime(start_from)

    test_date_split = start_from - pd.Timedelta(test_days, unit='D')
    train_date_msk = date_data < test_date_split
    
    return train_date_msk, ~train_date_msk


