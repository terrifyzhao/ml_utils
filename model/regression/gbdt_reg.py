from sklearn.ensemble import GradientBoostingRegressor
from config.base_config import *
from model.grid_search import grid_search_reg


def model(x, y, params):
    best_params = grid_search_reg(x, y, GradientBoostingRegressor(), params)
    learning_rate = best_params['learning_rate']
    subsample = best_params['subsample']
    max_depth = best_params['max_depth']
    n_estimators = best_params['n_estimators']
    cls = GradientBoostingRegressor(learning_rate=learning_rate,
                                    subsample=subsample,
                                    max_depth=max_depth,
                                    n_estimators=n_estimators,
                                    random_state=random_state)
    return cls
