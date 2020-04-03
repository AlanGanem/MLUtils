import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class RobustOneHotSparse:
    def __init__(self):
        return

    def listify_df(self, df):
        df = df.values.tolist()
        return df

    def stringify_columns(self, df, columns, sep):
        for column in columns:
            df.loc[:, column] = column + sep + df[column].astype(str)
        return df[columns]

    def undummerize_matrix(self, matrix):
        # values_dict = {col:[] for col in self.columns}
        # values = np.array(map(lambda x: x.split(sep).__getitem__(0), matrix.flatten()))
        # keys = np.array(map(lambda x: x.split(sep).__getitem__(1), matrix.flatten()))

        # for key,value in zip(keys,values):
        #     values_dict[key].append(value)

        # return values_dict
        raise NotImplementedError('inverse transform contain errors')

    def unstrigify_matrix(self, matrix, columns):
        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, columns=columns)
        return df

    def matrix_to_df(self, matrix, columns):
        matrix = np.array(matrix)
        df = pd.DataFrame(matrix, columns=columns)
        return df

    def settify_columns(self, columns):

        if columns.__class__ not in [list, tuple, set]:
            columns = [columns]
        if columns.__class__ != set:
            columns = set(columns)
        return columns

    def fit(self, df, columns, sep='__', dummy_na = True):
        df = df.copy()
        if dummy_na:
            df = df.append(pd.Series(), ignore_index=True)
            df = df.fillna('NaN')
        columns = self.settify_columns(columns)
        string_df = self.stringify_columns(df, columns, sep)
        df_list = self.listify_df(string_df)
        cv = CountVectorizer(tokenizer=self.tokenizer, lowercase=False, binary=True)
        cv.fit(df_list)

        self.count_vectorizer = cv
        self.categorical_features = columns
        # sorted list isntead of dict
        self.categorical_dummies = [k for k in sorted(self.count_vectorizer.vocabulary_, key=self.count_vectorizer.vocabulary_.get, reverse=False)]

        self.sep = sep
        return self

    def tokenizer(self,doc):
        return doc

    def fit_transform(self, df, columns, sep='__', return_df=True):
        df = df.copy()
        self.fit(df, columns, sep)
        transformed_result = self.transform(df, return_df)
        return transformed_result

    def assert_columns(self, df, columns):

        columns_not_in_df = [i for i in columns if i not in df]
        if columns_not_in_df:
            print('{} not in DataFrame. Columns will be created with ##EMPTY## label'.format(columns_not_in_df))
            for column in columns_not_in_df:
                df[column] = '##EMPTY##'
        return df

    def transform(self, df, return_df=True):
        df = df.copy()
        df = self.assert_columns(df, self.categorical_features)
        string_df = self.stringify_columns(df, self.categorical_features, self.sep)
        df_list = self.listify_df(string_df)
        transformed_result = self.count_vectorizer.transform(df_list)
        transformed_result = transformed_result.astype(np.int8)
        if return_df:
            dummy_df = self.matrix_to_df(transformed_result.A, self.categorical_dummies)
            transformed_result = pd.concat([df, dummy_df], axis=1)

        return transformed_result

    def inverse_transform(self):
        raise NotImplementedError('inverse transform contain errors')

