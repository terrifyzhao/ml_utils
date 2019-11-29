from sklearn.ensemble import GradientBoostingClassifier
from config.base_config import *


def model(params):
    learning_rate = params['learning_rate']
    subsample = params['subsample']
    max_depth = params['max_depth']
    n_estimators = params['n_estimators']
    cls = GradientBoostingClassifier(learning_rate=learning_rate,
                                     subsample=subsample,
                                     max_depth=max_depth,
                                     n_estimators=n_estimators,
                                     random_state=random_state)
    return cls
