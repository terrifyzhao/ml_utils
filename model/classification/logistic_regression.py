from sklearn.linear_model import LogisticRegression
from config.base_config import *


def model(params):
    C = params['C']
    penalty = params['penalty']
    cls = LogisticRegression(C=C, penalty=penalty, random_state=random_state, solver='liblinear')
    return cls
