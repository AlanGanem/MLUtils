# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:30:37 2019

@author: User Ambev
"""

import pandas as pd
import joblib

class RobustOneHotEncoder():
    
    @classmethod
    def load(cls, loading_path, **joblibargs):        
        return joblib.load(loading_path, **joblibargs)
    
    def save(self, saving_path, **joblibargs):        
        joblib.dump(self, saving_path, **joblibargs)


    def __init__(self):
        return
    
    def fit(self,X,cat_columns, prefix_sep = '__', sparse = False, dummy_na = True ,**getdummiesargs):
        df = X
        self.cat_columns = cat_columns
        self.prefix_sep = prefix_sep
        assert all([col in df.columns for col in self.cat_columns])
        
        print('applying pd.get_dummies method')
        one_hot_fit = pd.get_dummies(df, columns = cat_columns, prefix_sep = prefix_sep, sparse = sparse,dummy_na = dummy_na,**getdummiesargs)
        print('Done')
        
        self.cat_dummies = [col for col in one_hot_fit
               if prefix_sep in col 
               and col.split(prefix_sep)[0] in self.cat_columns]
        
        self.nested_cat_dummies = {cat:[] for cat in cat_columns}
        for dummy_cat in self.cat_dummies:
        	self.nested_cat_dummies[dummy_cat.split(self.prefix_sep)[0]].append(dummy_cat)

        self.n_cat_dummies = {cat:[] for cat in cat_columns}
        for cat in self.nested_cat_dummies:
        	self.n_cat_dummies[cat] = len(self.nested_cat_dummies[cat])

        	
        return self
    
    def transform(self, X, verbose = True, sparse = False, dummy_na = True, return_new_columns = False):
        df = X
        one_hot_transform = pd.get_dummies(df, prefix_sep=self.prefix_sep, 
                                   columns=self.cat_columns, sparse = sparse, dummy_na = dummy_na)
        
        # Remove additional columns
        for col in one_hot_transform.columns:
            if ("__" in col) and (col.split("__")[0] in self.cat_columns) and col not in self.cat_dummies:
                if verbose:
                    print("Removing additional feature {}".format(col))
                one_hot_transform.drop(col, axis=1, inplace=True)
                
        for col in self.cat_dummies:
            if col not in one_hot_transform.columns:
                if verbose:
                    print("Adding missing feature {}".format(col))
                one_hot_transform[col] = 0

        if not return_new_columns == True:
            return one_hot_transform
        else:
            return {'one_hot_transform':one_hot_transform, 'cat_dummies':self.cat_dummies, 'cat_dummies':self.nested_cat_dummies}
