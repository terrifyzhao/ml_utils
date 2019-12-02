# ml_utils

ml_utils is a AutoML tool, provide an input CSV and a target label to predict, auto generate a model 

# how to use
```
python pipeline.py -n='name.csv' -l='label'
```

# support model
+ LR
+ RandomForest
+ SVM
+ GBDT

# how to choose model
the `config/base_config.py` file has a param `model_name`, modify the param to choose your model

# how to tune model's params
modify `config/params_config.json` file


# model_path
`model/output/model_name_191130_1706.bin`

# how to predict
```python
import joblib

model=joblib.load('model_path')
# predict result
pred = model.predict('x_test')
# predict probability
prob = model.predict_proba('x_test')
```