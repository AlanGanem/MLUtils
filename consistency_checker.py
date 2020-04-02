import pandas as pd
import tqdm
import numpy as np
import difflib
import joblib


class ConsistencyChecker:

    @classmethod
    def load(cls, loading_path, **joblibargs):
        return joblib.load(loading_path, **joblibargs)

    def save(self, saving_path, **joblibargs):
        joblib.dump(self, saving_path, **joblibargs)

    def __init__(self):
        return

    def fit(self, df):
        self.columns = df.columns
        self.dtypes = df.dtypes
        self.description = self._get_description(df)
        self.dtypes = df.dtypes
        return self

    def check_values(self, check_df, absolute=True):
        return self._check_col_values(check_df, absolute)

    def check_types(self, check_df):
        names = self._check_col_names(check_df)
        types = self._check_col_types(check_df)
        return {'names': names, 'types': types}

    def _get_description(self, df):
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
        standard_columns = set(self.columns)
        check_columns = set(check_df.columns)

        matching_columns = standard_columns.intersection(check_columns)
        check_not_in_standard = check_columns - standard_columns
        standard_not_in_check = standard_columns - check_columns

        closest_matches_check = {}
        for col in standard_not_in_check:
            try:
                closest_matches_check[col] = difflib.get_close_matches(col, list(check_columns), n=1, cutoff=0.0)[0]
            except:
                print('No match found for {}'.format(col))
                closest_matches_check[col] = None

        check_dict = {
            'matching': matching_columns,
            'check_not_in_standard': check_not_in_standard,
            'standard_not_in_check': standard_not_in_check,
            'closest_matches': closest_matches_check
        }

        return check_dict

    def _check_col_types(self, check_df):
        # check dtypes
        matching_columns = set(self.columns).intersection(set(check_df.columns))
        dtypes_check = self.dtypes[matching_columns] == check_df.dtypes[matching_columns]
        dtypes_check_dict = {}
        for col in dtypes_check.index:
            dtypes_check_dict[col] = (dtypes_check[col], (self.dtypes[col], check_df.dtypes[col]))
        return dtypes_check_dict

    def _check_col_values(self, check_df, absolute):
        description_dict_check = self._get_description(check_df)
        description_dict_standard = self.description
        dtypes_check_dict = self._check_col_types(check_df)
        matching_columns = set(check_df.columns).intersection(set(self.columns))

        matching_names_and_types = [i for i in matching_columns if dtypes_check_dict[i][0] == True]
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
            else:
                check_values_dict[col] = self._check_cat_values(description_dict_check, col)
                if absolute == False:
                    check_values_dict[col] = {key: check_values_dict[col][key] / check_values_dict[col]['len_standard']
                                              for key in check_values_dict[col]}

        return check_values_dict

    def _check_cat_values(self, description_dict_check, col):

        set1 = set(self.description[col]['description']['unique'])
        set2 = set(description_dict_check[col]['description']['unique'])
        intrsc = set1.intersection(set2)
        values_dict = {'len_standard': len(set1), 'len_check': len(set2), 'intersection': len(intrsc)}
        return values_dict




