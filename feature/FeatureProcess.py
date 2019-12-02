#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/11/29
# @Author : 胡茂海
# @Site   : 
# @File   : FeatureProcess.py

from sklearn import preprocessing as skp
import Features as ft


class FeaturesStandard(object):
    def __init__(self, method='StandardScaler'):
        self.extra = False
        try:
            self.method = eval(f"skp.{method}()")
        except Exception as e:
            self.extra = True
        if self.extra:
            try:
                self.method = eval(f"ft.{method}()")
            except Exception as e:
                raise AttributeError('传入的方法名不对，无法实例化对象')

    def fit(self, *args):
        return self.method.fit(*args)

    def transform(self, *args):
        return self.method.transform(*args)
