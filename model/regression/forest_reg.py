from sklearn.ensemble import RandomForestRegressor
from config.base_config import *


def model(params):
    # "gini" for the Gini impurity and "entropy" for the information gain.
    criterion = params['criterion']
    max_depth = params['max_depth']
    n_estimators = params['n_estimators']
    cls = RandomForestRegressor(criterion=criterion,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                random_state=random_state)
    return cls
