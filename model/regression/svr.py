from sklearn.svm import SVR


def model(params):
    # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    kernel = params['kernel']
    C = params['C']
    cls = SVR(kernel=kernel, C=C)
    return cls
