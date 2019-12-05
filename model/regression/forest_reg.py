from sklearn.ensemble import RandomForestRegressor
from config.base_config import *
from model.grid_search import grid_search_reg


def model(x, y, params):
    best_params = grid_search_reg(x, y, RandomForestRegressor(), params)
    # criterion = best_params['criterion']
    max_depth = best_params['max_depth']
    n_estimators = best_params['n_estimators']
    cls = RandomForestRegressor(
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                random_state=random_state)
    return cls
