import numpy as np


class MissvalueFilter(object):

    def __init__(self, method='filter', missing_values=np.nan, threshold= 0.85):
        self.missing_values= missing_values
        self.threshold=threshold

    def fit(self, X, y=None):
        self.index = np.isnan(X).sum(axis=0)/X.shape[0] > self.threshold

    def transform(self, X):
        return X[:, ~self.index]

