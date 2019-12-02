import json
from sklearn.model_selection import train_test_split
from config.base_config import *
import time
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
import joblib


def save_model(model, model_name):
    import time
    try:
        time = time.strftime('%y%m%d_%H%M')
        path = 'output/' + model_name + '_' + time + '.bin'
        joblib.dump(model, path)
        print('save model success, model name:', model_name + '_' + time + '.bin')
    except:
        print('save model fail')


def train(X, y):
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,
                                                        test_size=split_size,
                                                        random_state=random_state)
    json_str = json.load(open('../config/params_config.json', encoding='utf-8'))

    model_dic = {}
    for j in json_str:
        name = j['model_name']
        param = j['params']
        model_dic[name] = param

    for name in model_name:
        params = model_dic[name]

        model = None

        if name == 'logistic_regression':
            from model.classification.logistic_regression import model
        elif name == 'svc':
            from model.classification.svc import model
        elif name == 'forest_cls':
            from model.classification.forest_cls import model
        elif name == 'gbdt_cls':
            from model.classification.gbdt_cls import model
        elif name == 'lightgbm_cls':
            from model.classification.lightgbm_cls import model
        elif name == 'xgboost_cls':
            from model.classification.xgboost_cls import model
        elif name == 'linear_regression':
            from model.regression.linear_regression import model
        elif name == 'svr':
            from model.regression.svr import model
        elif name == 'forest_reg':
            from model.regression.forest_reg import model
        elif name == 'gbdt_reg':
            from model.regression.gbdt_reg import model
        elif name == 'lightgbm_reg':
            from model.regression.lightgbm_reg import model
        elif name == 'xgboost_reg':
            from model.regression.xgboost_reg import model

        cls = model(X_train, y_train, params)
        save_model(cls, name)
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
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data['data']
    y = data['target']

    train(X, y)

