from sklearn import impute
import preprocess as pre
from utils.util import back_args_str


class PreprocessMissvalue(object):
    def __init__(self, method='SimpleImputer', *args, **kwargs):
        '''
        :param method:
        SimpleImputer
        parameters for SimpleImputer:
            - strategy: mean, median, constant, most_frequent
            - fill_value: When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
            - add_indicator: add binary indicators for missing values
        '''
        self.extra = False
        try:
            self.method = eval(f"impute.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"pre.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)
