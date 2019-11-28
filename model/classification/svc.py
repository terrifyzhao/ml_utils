from sklearn.svm import SVC
from config.base_config import *


def model(params):
    # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    kernel = params['kernel']
    C = params['C']
    cls = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
    return cls
