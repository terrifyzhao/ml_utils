from sklearn.linear_model import LogisticRegression


def model(params):
    c = params['c']
    random_state = params['random_state']
    penalty = params['penalty']
    cls = LogisticRegression(C=c, random_state=random_state, penalty=penalty)
    return cls
