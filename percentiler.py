import numpy as np
import pandas as pd
from tqdm import tqdm



class Percentiler:
    '''
    fits an array and determines to which percentile a new item belongs to based on the fitted array
    '''

    def __init__(self):
        return

    def fit(self, array):
        array = np.array(array)
        assert (array.shape[0] == array.flatten().shape[0])        
    
        array = array.flatten()
        temp = array.argsort()
        self.percentile = np.arange(len(array))[temp.argsort()]/(len(array)-1)
        self.array = array
        return self

    def transform(self, x):
        idx = self._find_nearest(self.array, x)
        try:
            value = self.percentile[idx]
            try:
                value[np.isnan(x)] = np.nan
            except:
                if np.isnan(x):
                    value = np.nan
            return value
        
        except:
            return np.nan

    def fit_transform_item(self, array, item=-1):
        array = np.array(array)
        self.fit(array)
        return self.transform(array[item])

    def inverse_transform(self, percentile):
        return np.percentile(self.array, 100*percentile, axis = -1)
    
    def _find_nearest(self, array, value):
        array = np.asarray(array)
        value = np.array(value)
        try:
            idx = np.array([(np.abs(array - i)).argmin() for i in value])
        except:
            idx = (np.abs(array - value)).argmin()
                
        return idx

   
class GroupPercentiler:
    
    def __init__(self):
        return
    
    def fit(self,
            data,
            group_by,
            observed_variable,
            date_index,
            window_size = '30D',
            min_samples = 30,
           ):

        online_estimators_dict = {}
        for cluster in tqdm(data[group_by].unique()):
            pct = Percentiler()
            subset_df = data[data[group_by] == cluster].set_index(date_index)[observed_variable].dropna()
            if len(subset_df.last(window_size)) > min_samples:
                array = subset_df.last(window_size)        
            else:
                array = subset_df.iloc[-min_samples:]
            pct.fit(array)
            online_estimators_dict[cluster] = pct
        
        self.online_estimators_dict = online_estimators_dict
        self.group_by = group_by
        self.observed_variable = observed_variable
        self.date_index = date_index
        
        return self
    
    def transform(self,data):
        data = data.copy()
        data['_PERCENTILE'] = np.nan
        for cluster in tqdm(data[self.group_by].unique()):
            msk = data[self.group_by] == cluster
            
            values = data[msk][self.observed_variable]
            percentiles = self.online_estimators_dict[cluster].transform(values)
            data.loc[msk, '_PERCENTILE'] = percentiles            
        return data
    
    def inverse_transform(self, data, percentile_col):
        data = data.copy()
        data['_INV_PERCENTILE'] = np.nan
        for cluster in tqdm(data[self.group_by].unique()):
            msk = data[self.group_by] == cluster
            values = data[msk][percentile_col]
            percentiles = self.online_estimators_dict[cluster].inverse_transform(values)
            data.loc[msk, '_INV_PERCENTILE'] = percentiles            
        return data
 
#usage example
#gpct = GroupPercentiler()
#gpct.fit(data)
#gpct.transform(data.sample(1000))
