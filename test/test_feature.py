from FeatureProcess import *
import pandas as pd
import numpy as np

fsd = FeaturesStandard()
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scr=fsd.fit(data)
print(scr.mean_)
print(scr.transform(data))

print('--------------------')


fe = FeaturesEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc =fe.fit(X)
print(enc.categories_)
print(enc.transform([['Female', 1], ['Male', 4]]).toarray())
print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
print(enc.get_feature_names(['gender', 'group']))


fd = FeaturesDecomposition(n_components=2)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
p = fd.fit(X)
print(p.explained_variance_ratio_)
print(p.singular_values_)

fs = FeaturesSelection(threshold=(.8 * (1 - .8)))
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
fs.fit(X)
X = fs.transform(X)
print(X)


