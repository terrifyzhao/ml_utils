import json
from sklearn.model_selection import train_test_split
from config.base_config import *
import time
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, mean_squared_error
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


def save_model(model, model_name):
    import time
    try:
        time = time.strftime('%y%m%d_%H%M')
        path = 'output/' + model_name + '_' + time + '.bin'
        joblib.dump(model, path)
        print('save model success, model name:', model_name + '_' + time + '.bin')
    except:
        print('save model fail')


def ensemble_model_mean(y_prob):
    y_prob = np.array(y_prob)
    res = np.mean(np.array(y_prob), axis=0)
    return res


def ensemble_model_lr(x, y, dtype):
    if dtype == 'cls':
        model = LogisticRegression(random_state=random_state, solver='liblinear')
    elif dtype == 'reg':
        model = LinearRegression()
    model.fit(x, y)
    return model


def calculate_score_cls(y_true, y_prob):
    y_pred = np.argmax(y_prob, 1).astype('int')
    y_prob = [i[1] for i in y_prob]

    acc = accuracy_score(y_true, y_pred)
    print('acc: ', acc)

    recall = recall_score(y_true, y_pred)
    print('recall: ', recall)

    f1 = f1_score(y_true, y_pred)
    print('f1: ', f1)

    auc = roc_auc_score(y_true, y_prob)
    print('auc: ', auc)

    return acc, recall, f1, auc


def calculate_score_reg(y_true, y_pred):
    y_true = np.array(y_true)
    loss = mean_squared_error(y_true, y_pred)
    print('loss:', loss)

    return loss


def train(X, y, dtype):
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,
                                                        test_size=split_size,
                                                        random_state=random_state)
    json_str = json.load(open(params_config_path, encoding='utf-8'))

    model_dic = {}
    for j in json_str:
        name = j['model_name']
        param = j['params']
        model_dic[name] = param

    if dtype == 'cls':
        cls_model(model_dic, X_train, X_eval, y_train, y_eval)
    elif dtype == 'reg':
        reg_model(model_dic, X_train, X_eval, y_train, y_eval)


def cls_model(model_dic, X_train, X_eval, y_train, y_eval):
    df = pd.DataFrame(columns=['name', 'acc', 'recall', 'f1', 'auc'])
    y_prob_list, score_list = [], []
    ensemble_x = []
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

        cls = model(X_train, y_train, params)
        print('model: ', cls)
        start = time.time()
        print('train begin')
        cls.fit(X_train, y_train)
        print('train time: ', time.time() - start)
        save_model(cls, name)

        ensemble_x.append([i[1] for i in cls.predict_proba(X_train)])

        y_prob = cls.predict_proba(X_eval)
        score = calculate_score_cls(y_eval, y_prob)

        score_list.append(score)
        y_prob_list.append(y_prob)

    ensemble_model = ensemble_model_lr(np.array(ensemble_x).T, y_train, 'cls')
    print()
    print('*' * 100)
    print('ensemble_model_lr')
    ensemble_lr_prob = ensemble_model.predict_proba(np.array(y_prob_list)[:, :, 1].T)
    ensemble_score = calculate_score_cls(y_eval, ensemble_lr_prob)
    score_list.append(ensemble_score)

    ensemble_prob = ensemble_model_mean(y_prob_list)
    print()
    print('*' * 100)
    print('ensemble_model_mean')
    ensemble_score = calculate_score_cls(y_eval, ensemble_prob)
    score_list.append(ensemble_score)

    df['name'] = model_name + ['ensemble_model_lr'] + ['ensemble_model_mean']
    score_list = np.array(score_list)
    for i, c in enumerate(['acc', 'recall', 'f1', 'auc']):
        df[c] = score_list[:, i]
    df.to_csv('output/result.csv', encoding='utf_8_sig', index=False)


def reg_model(model_dic, X_train, X_eval, y_train, y_eval):
    df = pd.DataFrame(columns=['name', 'loss'])
    y_pred_list, score_list = [], []
    ensemable_x = []
    for name in model_name:
        params = model_dic[name]

        model = None

        if name == 'linear_regression':
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
        print('model: ', cls)
        start = time.time()
        print('train begin')
        cls.fit(X_train, y_train)
        print('train time: ', time.time() - start)
        save_model(cls, name)

        ensemable_x.append(cls.predict(X_train))
        y_pred = cls.predict(X_eval)
        y_pred_list.append(y_pred)
        score = calculate_score_reg(y_eval, y_pred)
        score_list.append(score)

    ensemble_model = ensemble_model_lr(np.array(ensemable_x).T, y_train, 'reg')
    print()
    print('*' * 100)
    print('ensemble_model_lr')
    ensemble_pred = ensemble_model.predict(np.array(y_pred_list).T)
    ensemble_score = calculate_score_reg(y_eval, ensemble_pred)
    score_list.append(ensemble_score)

    ensemble_pred = ensemble_model_mean(y_pred_list)
    print()
    print('*' * 100)
    print('ensemble_model_mean')
    ensemble_score = calculate_score_reg(y_eval, ensemble_pred)
    score_list.append(ensemble_score)

    df['name'] = model_name + ['ensemble_model_lr'] + ['ensemble_model_mean']
    df['loss'] = score_list
    df.to_csv('output/result.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    cls = 0
    if cls:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        X = data['data']
        y = data['target']
        train(X, y, dtype='cls')
    else:
        from sklearn.datasets import load_boston

        data = load_boston()
        X = data['data']
        y = data['target']

        train(X, y, dtype='reg')
