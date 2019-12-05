import numpy as np


class PreprocessOutlier(object):
    def __init__(self, strategy="filter", fill_value=np.nan, thresh=3.5):
        '''
        :param strategy: "filter", "replace"
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

        if self.strategy == 'replace':
            X = X.astype("float")
            X[mask] = self.fill_value
        else:
            mask_row = np.sum(1 - mask, axis=1) == X.shape[1]
            X = X[mask_row]

        return X
