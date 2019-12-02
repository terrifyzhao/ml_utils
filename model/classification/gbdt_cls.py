from sklearn.ensemble import GradientBoostingClassifier
from config.base_config import *
from model.grid_search import grid_search


def model(x, y, params):
    best_params = grid_search(x, y, GradientBoostingClassifier(), params)
    learning_rate = best_params['learning_rate']
    subsample = best_params['subsample']
    max_depth = best_params['max_depth']
    n_estimators = best_params['n_estimators']
    cls = GradientBoostingClassifier(learning_rate=learning_rate,
                                     subsample=subsample,
                                     max_depth=max_depth,
                                     n_estimators=n_estimators,
                                     random_state=random_state)
    return cls
