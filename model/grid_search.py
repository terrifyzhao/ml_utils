from sklearn.model_selection import GridSearchCV
from config.base_config import *


def grid_search_cls(x, y, model, params):
    grid = GridSearchCV(model, params, cv=5, verbose=verbose, scoring='roc_auc')
    print()
    print('*' * 100)
    print('grid search begin')
    res = grid.fit(x, y)
    print('best auc:', res.best_score_)
    print('best param:', res.best_params_)
    return res.best_params_


def grid_search_reg(x, y, model, params):
    grid = GridSearchCV(model, params, cv=5, verbose=verbose, scoring='neg_mean_squared_error', iid=True)
    print()
    print('*' * 100)
    print('grid search begin')
    res = grid.fit(x, y)
    print('best loss:', res.best_score_)
    print('best param:', res.best_params_)
    return res.best_params_
