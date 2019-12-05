#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/11/29
# @Author : 胡茂海
# @Site   : 
# @File   : FeatureProcess.py

from sklearn import preprocessing as skp
from sklearn import decomposition  as dpn
from sklearn import feature_selection as fsn
import Features as ft


def back_args_str(*args, **kwargs):
    largs = [f"'{str(a)}'" if isinstance(a, str) else str(a) for a in args]
    kw = [str(k) + '=' + ("'" + str(v) + "'" if isinstance(v, str) else str(v)) for k, v in kwargs.items()]
    largs.extend(kw)
    return ','.join(largs)


class FeaturesStandard(object):
    def __init__(self, method='StandardScaler', *args, **kwargs):
        self.extra = False
        try:
            self.method = eval(f"skp.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesEncoder(object):
    def __init__(self, method='OneHotEncoder',*args, **kwargs):
        self.extra = False
        try:
            self.method = eval(f"skp.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesDecomposition(object):
    def __init__(self, method='PCA',*args, **kwargs):
        self.extra = False
        try:
            self.method = eval(f"dpn.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesSelection(object):
    def __init__(self, method='VarianceThreshold',*args, **kwargs):
        self.extra = False
        try:
            self.method = eval(f"fsn.{method}({back_args_str(*args, **kwargs)})")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{method}({back_args_str(*args, **kwargs)})")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)




if __name__ == '__main__':
    pass