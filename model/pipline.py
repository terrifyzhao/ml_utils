import json
from sklearn.model_selection import train_test_split
import pandas as pd
from ..config.base_config import *

df = pd.read_csv('')
X = df.drop(['label'], axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=split_size,
                                                    random_state=random_state)

json_str = json.load(open('model_config.json', encoding='utf-8'))
for j in json_str:
    name = j['model_name']
    param = j['prams']
    cls = None
    if name == 'logistic_regression':
        from .logistic_regression import model

        cls = model(param)
    cls.fit(X_train, y_train)

