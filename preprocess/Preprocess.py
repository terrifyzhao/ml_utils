'''

'''
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
import fancyimpute as fyi
import preprocess.preprocess_expand as exp
from utils.util import back_args_str, auto_pate


class PreprocessMissvalue(object):
    def __init__(self, method='SimpleImputer()'):
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
            self.method = eval(f"impute.{auto_pate(method)}")
        except Exception as e:
            self.extra = True
        if self.extra:
            self.extra = False
            try:
                self.method = eval(f"fyi.{auto_pate(method)}")
            except Exception as e:
                self.extra = True

            if self.extra:
                try:
                    self.method = eval(f"exp.{auto_pate(method)}")
                except Exception as e:
                    raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class PreprocessOutlier(object):
    def __init__(self, method='Replace()'):
        try:
            self.method = eval(f"exp.{auto_pate(method)}")
        except Exception as e:
            raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)

