from sklearn.linear_model import LogisticRegression
from config.base_config import *
from model.grid_search import grid_search_cls


def model(x, y, params):
    best_params = grid_search_cls(x, y, LogisticRegression(solver='liblinear'), params)
    C = best_params['C']
    penalty = best_params['penalty']
    cls = LogisticRegression(C=C, penalty=penalty, random_state=random_state, solver='liblinear')
    return cls
