#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/11/29
# @Author : 胡茂海
# @Site   : 
# @File   : FeatureProcess.py

from sklearn import preprocessing as skp
from sklearn import decomposition  as dpn
from sklearn import feature_selection as fsn
import feature_expand as ft
from utils.util import back_args_str

def auto_pate(method):
    """自动添加括号"""
    method = str.strip(method)
    if method[-1]!=')':
        if '(' not in method:
            method = method+'()'
        else:
            method =  method+')'
    return method

class FeaturesStandard(object):
    def __init__(self, method="StandardScaler()"):
        self.extra = False
        try:
            self.method = eval(f"skp.{auto_pate(method)}")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{auto_pate(method)}")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesEncoder(object):
    def __init__(self, method="OneHotEncoder(handle_unknown='ignore')"):
        self.extra = False
        try:
            self.method = eval(f"skp.{auto_pate(method)}")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{auto_pate(method)}")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesDecomposition(object):
    def __init__(self, method="PCA(n_components=2)"):
        self.extra = False
        try:
            self.method = eval(f"dpn.{auto_pate(method)}")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{auto_pate(method)}")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)


class FeaturesSelection(object):
    def __init__(self, method="VarianceThreshold(threshold=0.16)"):
        self.extra = False
        try:
            self.method = eval(f"fsn.{auto_pate(method)}")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{auto_pate(method)}")
            except Exception as e:
                raise AttributeError('传入的方法名或者参数不对，无法实例化对象')

    def fit(self, *args,**kwargs):
        return self.method.fit(*args,**kwargs)

    def transform(self, *args,**kwargs):
        return self.method.transform(*args,**kwargs)




if __name__ == '__main__':
    pass