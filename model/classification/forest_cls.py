from sklearn.ensemble import RandomForestClassifier
from config.base_config import *
from model.grid_search import grid_search


def model(x, y, params):
    best_params = grid_search(x, y, RandomForestClassifier(), params)
    # "gini" for the Gini impurity and "entropy" for the information gain.
    criterion = best_params['criterion']
    max_depth = best_params['max_depth']
    n_estimators = best_params['n_estimators']
    cls = RandomForestClassifier(criterion=criterion,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 random_state=random_state)
    return cls
