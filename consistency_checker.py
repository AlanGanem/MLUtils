import pandas as pd
import tqdm
import numpy as np
import difflib
import joblib


class ConsistencyChecker:
    '''
    A class for checking consistency (column names, dtypes and values) of two dataframes, following the 'fit-check'
    paradigm
    '''

    @staticmethod
    def check_identical(df1, df2):
        '''
        cehcks wheter two dataframes are identical
        :param df1: first data frame
        :param df2: second data frame
        :return: returns a boolean series of identical columns
        '''
        return (df1 == df2).all()

    @staticmethod
    def check_col_distribution(df1, df2):
        '''
        compares distribution of 'column' in df1 and df2. Works for categorical and continuous columns
        :param self:
        :param check_df:
        :param column:
        :return:
        '''
        raise NotImplementedError

    @classmethod
    def load(cls, loading_path, **joblibargs):
        '''
        loads serialized consistency_checker object
        :param loading_path: path containing serialized object
        :param joblibargs: joblib.load kwargs
        :return: returns deserialized object
        '''
        return joblib.load(loading_path, **joblibargs)

    def save(self, saving_path, **joblibargs):
        '''
        serializes and saves consistency_checker object to saving_path
        :param saving_path: saving path of the file
        :param joblibargs: joblib.dump kwargs
        :return:
        '''
        joblib.dump(self, saving_path, **joblibargs)

    def __init__(self):
        '''
        Object instatiation
        '''
        return

    def fit(self, df):
        '''
        saves standard_df attributes do object state
        :param df: standard_df
        :return: self
        '''
        self.columns = df.columns
        self.dtypes = df.dtypes
        self.description = self._get_description(df)
        self.dtypes = df.dtypes
        return self

    def fast_check_names(self, check_df):
        '''
        checks if there's some column name mismatch between check_df and standard_df
        :param check_df: df to perform check
        :return: check boolean
        '''
        if not self.check_names(check_df)['in_both'].all():
            print(
                'Missmatching column names found!\nMake sure you are comparing all wanted columns checking obj.check_\
                names(df) (FUTURE WARNING)')
        return self.check_names(check_df)['in_both'].all()

    def fast_check_types(self, check_df):
        '''
        checks if there's some type mismatch between columns of check_df_ and standard_df
        :param check_df: df to perform check
        :return: check boolean
        '''
        if not self.check_types(check_df)['check'].all():
            print(
                'Missmatching column types found!\nMake sure you are comparing all wanted columns checking obj.check \
                _types(df) (FUTURE WARNING)')
        return self.check_types(check_df)['check'].all()

    def check_col_values(self, check_df, column, check_missing = False):
        '''
        Returns the unique values in check_df[column] that are not in standard_df[column] (unseen).
        If check_missing == True, returns also the values of standard_df[column] unseen in check_df[column]

        :param check_df: df to check values
        :param column: column to check
        :param check_missing: whether to check missing values in check_df
        :return:
        '''

        set_check = set(self._get_description(check_df[[column]])[column]['unique'])
        set_standard = set(self.description[column]['unique'])

        new_in_check = set_check - set_standard
        total_new_in_check_proportion = check_df[column].isin(new_in_check).mean()
        check_dict = {'unseen_values':new_in_check, 'unseen_proportion':total_new_in_check_proportion}

        if check_missing:
            not_in_check = set_standard - set_check
            unique_not_in_check_proportion = len(not_in_check)/len(set_standard)
            check_dict = {**check_dict, **{'missing_values':not_in_check,
                                           'missing_proportion':unique_not_in_check_proportion}}
        return check_dict

    def check_values(self, check_df, absolute=True, return_dict=False):
        '''
        compares check_df values to standard_df values, only of columns with matching names and types.
        if columns dtype == float, the values returned are a sustraction of both dfs 'description' method of pandas
        for descriptive statistics. if not, the values are compared as categorical, returning unique values for each
        column of each df and their intersection.
        absolute defines if whether the values are absolute values or are divided by de standard_df's values for each
        statistic.

        :param check_df:
        :param absolute:
        :param return_dict:
        :return: dict or dataframe of comparisons
        '''
        checker = self._check_col_values(check_df, absolute)
        if not return_dict:
            checker = pd.DataFrame(checker).T
            checker = checker.reset_index(drop=False).set_index(['type', 'index']).sort_index()
        return checker

    def check_types(self, check_df, return_dict=False):
        '''
        performs dtype checks for columns with matching names in check_df and standard_df
        :param check_df:
        :param return_dict:
        :return: returns a dict/dataframe containing boolean dtype checks and
        a tupple containing (standard_type, check_type)
        '''
        checker = self._check_col_types(check_df)
        if not return_dict:
            checker = pd.DataFrame(checker).T
        return checker

    def check_names(self, check_df, return_dict=False):
        '''
        checks columns names in check_df according to standard_df.
        returns name boolean checks and closest matches in case of name mismatches
        according to difflib.get_close_matches

        :param check_df:
        :param return_dict: whether to return a dict or a pandas DataFrame
        :return:
        '''
        checker = pd.DataFrame(self._check_col_names(check_df)).T
        return checker

    def _get_description(self, df):
        '''
        internal method to get df description
        :param df:
        :return:
        '''

        description_dict = {}
        for col in df:
            if not df[col].dtype in ['float']:
                description_dict[col] = {
                    'dtype': df[col].dtype,
                    'description': {
                        'nunique': df[col].nunique(),
                        'unique': list(df[col].unique())
                    }

                }
            else:
                description_dict[col] = {
                    'dtype': df[col].dtype,
                    'description': df[col].describe().to_dict()
                }

        return description_dict

    def _check_col_names(self, check_df):
        '''
        internal method to check col names in check_df
        :param check_df:
        :return:
        '''
        union_columns = set(self.columns).union(set(check_df.columns))
        check_dict = {}
        for col in union_columns:
            check_dict[col] = {}
            check_dict[col]['in_standard'] = col in self.columns
            check_dict[col]['in_check'] = col in check_df.columns
            check_dict[col]['in_both'] = check_dict[col]['in_standard'] and check_dict[col]['in_check']
            check_dict[col]['closest_match'] = np.nan
            check_dict[col]['closest_match_score'] = np.nan
            if not check_dict[col]['in_both']:
                if check_dict[col]['in_check']:
                    check_dict[col]['closest_match'] = \
                    difflib.get_close_matches(col, list(self.columns), n=1, cutoff=0.0)[0]
                else:
                    check_dict[col]['closest_match'] = \
                    difflib.get_close_matches(col, list(check_df.columns), n=1, cutoff=0.0)[0]
                check_dict[col]['closest_match_score'] = difflib.SequenceMatcher(None, col, check_dict[col][
                    'closest_match']).ratio()
        return check_dict

    def _check_col_types(self, check_df):
        '''
        Internal method to check check_df's columns dtypes
        :param check_df:
        :return:
        '''
        # check if there are names inncosistency
        self.fast_check_names(check_df)
        #
        matching_columns = set(self.columns).intersection(set(check_df.columns))
        dtypes_check = self.dtypes[matching_columns] == check_df.dtypes[matching_columns]
        dtypes_check_dict = {}
        for col in dtypes_check.index:
            dtypes_check_dict[col] = {'check': dtypes_check[col], 'types': (self.dtypes[col], check_df.dtypes[col])}
        return dtypes_check_dict

    def _check_col_values(self, check_df, absolute):
        '''
        Internal method to check check_df's col values
        :param check_df:
        :param absolute:
        :return:
        '''
        # check if there are names or types inncosistency
        self.fast_check_names(check_df)
        self.fast_check_types(check_df)
        #
        description_dict_check = self._get_description(check_df)
        description_dict_standard = self.description
        dtypes_check_dict = self._check_col_types(check_df)
        matching_columns = set(check_df.columns).intersection(set(self.columns))

        matching_names_and_types = [i for i in matching_columns if dtypes_check_dict[i]['check'] == True]
        check_values_dict = {}
        for col in matching_names_and_types:

            if description_dict_standard[col]['dtype'] == float:
                check_values_dict[col] = ((
                        pd.Series(description_dict_standard[col]['description']) - \
                        pd.Series(description_dict_check[col]['description'])))

                if absolute == False:
                    check_values_dict[col] = check_values_dict[col] / pd.Series(
                        description_dict_standard[col]['description'])

                check_values_dict[col] = check_values_dict[col].to_dict()
                check_values_dict[col]['type'] = dtypes_check_dict[col]['types'][0]
            else:
                check_values_dict[col] = self._check_cat_values(description_dict_check, col)
                if absolute == False:
                    check_values_dict[col] = {
                        key: check_values_dict[col][key] / check_values_dict[col]['unique_standard'] for key in
                        check_values_dict[col]}
                check_values_dict[col]['type'] = dtypes_check_dict[col]['types'][0]

        return check_values_dict

    def _check_cat_values(self, description_dict_check, col):
        '''
        Internal method to compare categorical values in check_df[col]
        :param description_dict_check:
        :param col:
        :return:
        '''

        set1 = set(self.description[col]['description']['unique'])
        set2 = set(description_dict_check[col]['description']['unique'])
        intrsc = set1.intersection(set2)
        new = set2 - intrsc
        values_dict = {'unique_standard': len(set1), 'unique_check': len(set2), 'intersection': len(intrsc),
                       'new_instances': list(new)}
        return values_dict