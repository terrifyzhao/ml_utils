from sklearn.svm import SVC
from config.base_config import *
from model.grid_search import grid_search


def model(x, y, params):
    best_params = grid_search(x, y, SVC(), params)
    # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    kernel = best_params['kernel']
    C = best_params['C']
    cls = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
    return cls
