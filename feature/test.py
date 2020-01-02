#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/11/29
# @Author : 胡茂海
# @Site   : 
# @File   : test.py


from FeatureProcess import *
import pandas as pd
import numpy as np

# data = pd.read_csv('titanic.csv')
# y = data['Survived']
# x = data[[c for c in data.columns if c !='Survived']]
# print(x)


fsd = FeaturesStandard('StandardScaler')
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scr=fsd.fit(data)
print(scr.mean_)
print(scr.transform(data))
print('--------------------')


data =pd.read_csv('../data/titanic.csv')
x = data['Age'].values
y = data['Survived'].values
fsd = FeaturesStandard('CutBins')
scr=fsd.fit(y,x)
print(scr.transform(x))
print('--------------------')



fe = FeaturesEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc =fe.fit(X)
print(enc.categories_)
print(enc.transform([['Female', 1], ['Male', 4]]).toarray())
print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
print(enc.get_feature_names(['gender', 'group']))


fd = FeaturesDecomposition()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
p = fd.fit(X)
print(p.explained_variance_ratio_)
print(p.singular_values_)

fs = FeaturesSelection()
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
fs.fit(X)
X = fs.transform(X)
print(X)


