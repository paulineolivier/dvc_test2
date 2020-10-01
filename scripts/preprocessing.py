import pandas as ps
from sklearn.base import TransformerMixin
from sklearn import preprocessing, impute
import numpy as np


# copy
class MakeCopyTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, df, y=None):
        return df.copy()

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df.copy()


# encoder
class LabelEncoderCol(TransformerMixin):
    def __init__(self, col):
        self.col = col
        self.prep = preprocessing.LabelEncoder()

    def fit_transform(self, df, y=None):
        self.prep.fit(df[self.col])
        df[self.col] = self.prep.transform(df[self.col])
        return df

    def fit(self, df, y=None):
        return self.prep.fit(df[self.col])

    def transform(self, df):
        df[self.col] = self.prep.transform(df[self.col])
        return df


# dummies
class DummiesVariable(TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit_transform(self, df, y=None):
        df_dummied = ps.get_dummies(df[self.col], prefix=self.col, drop_first=False)
        df = ps.concat([df, df_dummied], axis=1)
        return df

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df_dummied = ps.get_dummies(df[self.col], prefix=self.col, drop_first=False)
        df = ps.concat([df, df_dummied], axis=1)
        return df


# imputer
class SimpleImputerCol(TransformerMixin):
    def __init__(self, col, strategy):
        self.col = col
        self.imput = impute.SimpleImputer(missing_values=np.nan, strategy=strategy)

    def fit_transform(self, df, y=None):
        self.imput.fit(df[self.col].values.reshape(-1,1))
        df[self.col] = self.imput.transform(df[self.col].values.reshape(-1,1))
        return df

    def fit(self, df, y=None):
        return self.imput.fit(df[self.col].values.reshape(-1,1))

    def transform(self, df):
        df[self.col] = self.imput.transform(df[self.col].values.reshape(-1,1))
        return df


# drop columns
class DropColumnsTransformer(TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit_transform(self, df, y=None):
        return df.drop(self.columns_to_drop, axis=1, errors="ignore")

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df.drop(self.columns_to_drop, axis=1, errors="ignore")
