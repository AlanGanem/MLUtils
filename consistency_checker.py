import pandas as pd
import tqdm
import numpy as np
import difflib
import joblib


class ConsistencyChecker:

    @staticmethod
    def check_identical(df1, df2):
        return (check_df1 == check_df2).all()

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

    def fast_check_names(self, check_df):
        if not self.check_names(check_df)['in_both'].all():
            print(
                'Missmatching column names found!\nMake sure you are comparing all wanted columns checking obj.check_names(df) (FUTURE WARNING)')
        return

    def fast_check_types(self, check_df):
        if not self.check_types(check_df)['check'].all():
            print(
                'Missmatching column types found!\nMake sure you are comparing all wanted columns checking obj.check_types(df) (FUTURE WARNING)')
        return

    def check_values(self, check_df, absolute=True, return_dict=False):
        checker = self._check_col_values(check_df, absolute)
        if not return_dict:
            checker = pd.DataFrame(checker).T
            checker = checker.reset_index(drop=False).set_index(['type', 'index']).sort_index()
        return checker

    def check_types(self, check_df, return_dict=False):
        checker = self._check_col_types(check_df)
        if not return_dict:
            checker = pd.DataFrame(checker).T
        return checker

    def check_names(self, check_df, return_dict=False):
        checker = pd.DataFrame(self._check_col_names(check_df)).T
        return checker

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

        set1 = set(self.description[col]['description']['unique'])
        set2 = set(description_dict_check[col]['description']['unique'])
        intrsc = set1.intersection(set2)
        values_dict = {'unique_standard': len(set1), 'unique_check': len(set2), 'intersection': len(intrsc)}
        return values_dict