# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:44:49 2020

@author: User Ambev
"""
        

from scipy.sparse import csr_matrix, csc_matrix, dok_matrix
import tqdm        
import numpy as np
import pandas as pd
import networkx as nx

class PivotSparseDf():
    
    def get_link_defaults(self):
        
        def simple_agg(array, item):
                return 1
        
        def count_agg(array, item):
            return (array == item).sum()
        
        def freq_agg(array, item):
            return (array == item).mean()
        
        def mostfrequent_agg(array, item):
            if array.mode() == item:
                return 1
            else:
                return 0
        
        link_defaults = {
            'binary':simple_agg,
            'freq': freq_agg,
            'count': count_agg,
            'most_frequent': mostfrequent_agg
            }
        return link_defaults
    
    def __init__(self, index , columns, link_mode = 'binary'):
        
        if link_mode.__class__ == dict:
            if not set(link_mode) == set(columns):
                raise AssertionError('keys in link_mode must match columns. got {}, {} instead'.format(link_mode,columns))
        self.link_defaults = self.get_link_defaults()
        self.col_funcs = {col:{'groupping_function':{},'agg_function':{}} for col in columns}
        self.columns = columns
        self.index = index
        
        for col in columns:        
            
            if (link_mode.__class__ == str) or callable(link_mode):
                
                if link_mode == 'binary':
                    self.col_funcs[col]['groupping_function'] = set
                    self.col_funcs[col]['agg_function'] = self.link_defaults[link_mode]
                
                elif link_mode.__class__ == str:
                    self.col_funcs[col]['groupping_function'] = np.array
                    self.col_funcs[col]['agg_function'] = self.link_defaults[link_mode]
                
                elif callable(link_mode):
                    self.col_funcs[col]['groupping_function'] = np.array
                    self.col_funcs[col]['agg_function'] = link_mode
                    
            elif link_mode.__class__ == dict:        
                
                if link_mode[col] == 'binary':
                    self.col_funcs[col]['groupping_function'] = set
                    self.col_funcs[col]['agg_function'] = link_mode[col]
                
                elif link_mode[col].__class__ == str:
                    self.col_funcs[col]['groupping_function'] = np.array
                    self.col_funcs[col]['agg_function'] = link_mode[col]
                
                elif callable(link_mode[col]):
                    self.col_funcs[col]['groupping_function'] = np.array
                    self.col_funcs[col]['agg_function'] = link_mode[col]
                
                else:
                    raise TypeError('{} agg_function must be callable or str. got {} instead'.format(col, link_mode[col].__class__.__name__))
        
        
    def fit(self,data):
        '''
        performs pivoting operation with multiple columns and single index,
        returns a networkx-graph, sparse scipy-matrix or np-array.
        link_mode is a dictionary containing the aggregation method for multiple values in each cross_cell.
        A custom function may be passed. it has to contain only 2 parameters: array and item
        -array is the array containing the elements of the cross_cell
        -item is the item of the cross_cell on the loop (once the cross_cell is reached, the program will iterate over all items in the cell array)
        
        '''
        columns = self.columns
        index = self.index
        
        data = data.set_index(index)
        
        self.index_map = {i:v for i,v in enumerate(set(data.index))}
        self.index_inv_map = {v:i for i,v in enumerate(set(data.index))}
        
        self.cols_sets = {col:set(data[col]) for col in columns} 
        self.cols_map = {col:{i:v for i,v in enumerate(self.cols_sets[col])} for col in self.cols_sets}
        self.cols_inv_map = {col:{v:i for i,v in enumerate(self.cols_sets[col])} for col in self.cols_sets}
        
    def transform(self,data):
        
        columns = self.columns
        index = self.index
        data = data.set_index(index)
        
        groupped_data = {col:data.groupby(level = index)[col].apply(self.col_funcs[col]['groupping_function']) for col in columns}
        groupped_data = pd.DataFrame(groupped_data)
        
        matrix = dok_matrix((len(self.index_map),sum(len(self.cols_sets[col]) for col in columns)))
        
        i = 0
        for col in self.cols_sets:
            for idx in tqdm.tqdm(self.index_inv_map):
                try:
                    for item in groupped_data.loc[idx,col]:
                        if self.col_funcs[col]['agg_function'] == 'binary':
                            matrix[self.index_inv_map[idx],self.cols_inv_map[col][item]+i] = 1
                        else:
                            matrix[self.index_inv_map[idx],self.cols_inv_map[col][item]+i] = self.col_funcs[col]['agg_function'](array = groupped_data.loc[idx,col],item = item)
                except KeyError:
                    pass
                    
            i+=len(self.cols_sets[col])
        
        matrix = csr_matrix(matrix)
        
        return matrix
    