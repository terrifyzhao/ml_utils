'''

'''
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
import fancyimpute as fyi
import preprocess_expand as exp
from utils.util import back_args_str


class PreprocessMissvalue(object):
    def __init__(self, method='SimpleImputer', *args, **kwargs):
        '''
        :param method:
        1.SimpleImputer
        parameters for SimpleImputer:
            - strategy: mean, median, constant, most_frequent
            - fill_value: When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
            - add_indicator: add binary indicators for missing values

        2.IterativeImputer
        parameters for IterativeImputer:
            - max_iter: Maximum number of imputation rounds to perform before returning the imputations computed during the final round.
            - add_indicator: add binary indicators for missing values

        3.KNN
        parameters for KNN:
            - k : Number of neighboring rows to use for imputation.

        4.NuclearNormMinimization
        '''
        self.extra = False
        try:
            self.method = eval(f"impute.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"fyi.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')
            try:
                self.method = eval(f"exp.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class PreprocessOutlier(object):
    def __init__(self, strategy="filter", fill_value=np.nan, thresh=3.5):
        '''
        :param strategy: "filter", "replace"
        :param fill_value: when startegy="replace", replace outlier with np.nan
        '''
        self.strategy = strategy
        self.fill_value = fill_value
        self.thresh = thresh

    def fit(self, X, y=None):
        median = np.median(X, axis=0)
        diff = np.abs(X - median)
        med_abs_deviation = np.median(diff, axis=0)

        self.median = median
        self.med_abs_deviation = med_abs_deviation

        return self

    def transform(self, X):
        diff = np.abs(X - self.median)
        modified_z_score = 0.6745 * diff / self.med_abs_deviation

        mask = modified_z_score > self.thresh

        if self.strategy == 'replace':
            X = X.astype("float")
            X[mask] = self.fill_value
        else:
            mask_row = np.sum(1 - mask, axis=1) == X.shape[1]
            X = X[mask_row]

        return X


