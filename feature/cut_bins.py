#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/12/2
# @Author : 胡茂海
# @Site   : 
# @File   : cut_bins.py

import pandas as pd


def get_bins(Y,X,drop_ratio=1.0,n=10):
    total_sample = len(Y)
    df1 = pd.DataFrame({'X':X,'Y':Y})
    justmiss = df