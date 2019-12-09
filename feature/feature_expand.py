#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/11/29
# @Author : 胡茂海
# @Site   : 
# @File   : feature_expand.py

import pandas as pd
from feature.tools import bayesian_blocks

class CutBins(object):
    def __init__(self):
        self.bins = []
        self.labels = []

    def fit(self,x, bins=None, labels=None):
        """
        :param x:
        :param bins: 分割区间[0,10,18,35,55,120] 不指定，默认等频5等分
        :param labels: 区间名称['幼少','青春','成年','中年','老年']
        :return:
        """
        if bins:
            self.bins = bins
        else:
            self.bins = bayesian_blocks(x)
        if labels:
            self.labels =labels
        else:
            self.labels = range(len(bins))
        return self

    def transform(self,x):
        if not self.bins and not self.labels:
            raise AttributeError("instance is not fitted yet. Call 'fit' first")
        x = pd.cut(x, self.bins, labels=self.labels)
        return x