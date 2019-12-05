from sklearn.svm import SVR
from model.grid_search import grid_search_reg


def model(x, y, params):
    best_params = grid_search_reg(x, y, SVR(), params)
    # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    kernel = best_params['kernel']
    C = best_params['C']
    cls = SVR(kernel=kernel, C=C)
    return cls
