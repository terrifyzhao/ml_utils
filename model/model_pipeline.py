import json
from sklearn.model_selection import train_test_split
import pandas as pd
from config.base_config import *
import time
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score


def read_data(data_path, label_name):
    df = pd.read_csv(data_path)
    X = df.drop([label_name], axis=1).values
    y = df[label_name].values
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,
                                                        test_size=split_size,
                                                        random_state=random_state)
    return X_train, X_eval, y_train, y_eval


def test_data():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data['data']
    y = data['target']
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,
                                                        test_size=split_size,
                                                        random_state=random_state)
    return X_train, X_eval, y_train, y_eval


def train(data_path='', label_name=''):
    # X_train, X_eval, y_train, y_eval = read_data(data_path, label_name)
    X_train, X_eval, y_train, y_eval = test_data()
    json_str = json.load(open('../config/params_config.json', encoding='utf-8'))
    for j in json_str:
        name = j['model_name']
        param = j['params']
        model = None

        if name == 'logistic_regression':
            from model.classification.logistic_regression import model
        elif name == 'svc':
            from model.classification.svc import model
        elif name == 'forest_cls':
            from model.classification.forest_cls import model
        elif name == 'gbdt_cls':
            from model.classification.gbdt_cls import model
        elif name == 'linear_regression':
            from model.regression.linear_regression import model
        elif name == 'svr':
            from model.regression.svr import model
        elif name == 'forest_reg':
            from model.regression.forest_reg import model
        elif name == 'gbdt_reg':
            from model.regression.gbdt_reg import model

        cls = model(param)
        print()
        print('*' * 100)
        print('model: ', cls)
        start = time.time()
        print('start train')
        cls.fit(X_train, y_train)
        print('train time: ', time.time() - start)
        y_pred = cls.predict(X_eval)
        y_prob = [i[1] for i in cls.predict_proba(X_eval)]
        acc = accuracy_score(y_eval, y_pred)
        print('acc: ', acc)
        recall = recall_score(y_eval, y_pred)
        print('recall: ', recall)
        f1 = f1_score(y_eval, y_pred)
        print('f1: ', f1)
        auc = roc_auc_score(y_eval, y_prob)
        print('auc: ', auc)


if __name__ == '__main__':
    train()
