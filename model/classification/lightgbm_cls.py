from lightgbm import LGBMClassifier
from config.base_config import *
from model.grid_search import grid_search_cls


def model(x, y, params):
    best_params = grid_search_cls(x, y, LGBMClassifier(), params)
    learning_rate = best_params['learning_rate']
    num_leaves = best_params['num_leaves']
    subsample = best_params['subsample']
    colsample_bytree = best_params['colsample_bytree']
    max_depth = best_params['max_depth']
    n_estimators = best_params['n_estimators']
    cls = LGBMClassifier(learning_rate=learning_rate,
                         num_leaves=num_leaves,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         max_depth=max_depth,
                         n_estimators=n_estimators,
                         random_state=random_state)
    return cls
