'''

'''
import numpy as np


class Replace(object):
    def __init__(self, strategy="Replace", fill_value=np.nan, thresh=3.5):
        '''
        :param strategy:"Replace"
        :param fill_value: when startegy="replace", replace outlier with np.nan
        '''
        self.strategy = strategy
        self.fill_value = fill_value
        self.thresh = thresh

    def fit(self, X, y=None):
        median = np.median(X, axis=0)
        diff = np.abs(X - median)
        med_abs_deviation = np.median(diff, axis=0)

        self.median = median
        self.med_abs_deviation = med_abs_deviation

        return self

    def transform(self, X):
        diff = np.abs(X - self.median)
        modified_z_score = 0.6745 * diff / self.med_abs_deviation

        mask = modified_z_score > self.thresh

        X = X.astype("float")
        X[mask] = self.fill_value

        return X


class Filter(object):

    def __init__(self, method='filter', missing_values=np.nan, threshold= 0.85):
        '''
        :param method: 'filter' - filter columns
        :param missing_values:
        :param threshold:
        '''

        self.missing_values= missing_values
        self.threshold=threshold

    def fit(self, X, y=None):
        self.index = np.isnan(X).sum(axis=0)/X.shape[0] > self.threshold

    def transform(self, X):
        return X[:, ~self.index]

