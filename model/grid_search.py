from sklearn.model_selection import GridSearchCV


def grid_search(x, y, model, params):
    grid = GridSearchCV(model, params, cv=5, verbose=0, scoring='roc_auc')
    res = grid.fit(x, y)
    print()
    print('*'*100)
    print('grid search')
    print('auc:', res.best_score_)
    print('best param:', res.best_params_)
    return res.best_params_
