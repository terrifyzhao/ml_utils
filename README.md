# ml_utils

ml_utils是一个全自动的机器学习工具，只需要提供输入的csv文件与对应的label列名，即可自动选择模型，自动调参得到最终的模型。

# 如何使用
+ 提供数据，可放在`/data`目录下
+ 配置`config`目录下的参数文件，分别为`base_config`，`column_config`，`params_config`，每个文件的配置方法参考下文
+ 执行以下命令

```
python pipeline.py -n='name.csv' -l='label' -y='cls'
```

-n表示的是训练数据的路径，-l表示的是lable名字，-y表示的任务类型分类`cls`与回归`reg`

# 代码目录
![](https://raw.githubusercontent.com/terrifyzhao/ml_utils/master/dir.jpg)

+ config – 路径配置、特征配置、模型超参数配置
+ data – 训练数据
+ explore – 数据探索
+ feature – 特征工程
+ model – 分类、回归、超参数搜索
+ preprocess – 数据预处理
+ test – 模块测试
+ utils – 通用工具类

# 特征配置文件column_config
特征配置文件是一个csv文件，其格式如下

| features | types | outlier | missvalue | standard | encoder |
| --- | --- | --- | --- | --- | --- |
| Pclass | category |  |  |  |  |
| Sex | category |  |  |  | OrdinalEncoder |
| Age | numeric | Replace |  |  |  |
| Siblings/Spouses Aboard | category |  |  |  | OrdinalEncoder |
| Parents/Children Aboard | category |  |  |  |  |
| Fare | numeric |  | KNNImputer |  |  |

+ features：包含所有特征
+ types: 声明特征类型
+ outlier/missvalue/standard/encoder: 分别为四个模块，可针对每个特征提供相应的处理方法。未提供，则使用默认的处理方式。





# 支持的模型
+ LR
+ RandomForest
+ SVM
+ GBDT
+ lightgbm
+ xgboost

# 如何选择训练哪些模型base_config
`config/base_config.py`文件包含一个参数`model_name`，该参数是一个列表，修改该参数从而选择你需要训练的模型

# 如何调参params_config
修改 `config/params_config.json` 文件
逻辑回归：

+ C：正则化的惩罚项的倒数
+ penalty：正则化方法，可选值'l1', 'l2'

svm：

+ kernel：核函数，可选值'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
+ C：正则化的惩罚项的倒数

random forest：

+ criterion：测量分割质量的方法，可选值"gini", "entropy"
+ max_depth：树深度
+ n_estimators：树的个数

gbdt：

+ learning_rate：学习率
+ subsample：子模型数据的采样比例
+ max_depth：树深度
+ n_estimators：树的个数

lightgbm：

+ learning_rate：学习率
+ num_leaves：叶子节点的个数，
+ subsample：子模型数据的采样比例
+ colsample_bytree：子模型特征的采样比例
+ max_depth：树深度
+ n_estimators：树的个数


xgboost：

+ learning_rate：学习率
+ num_leaves：叶子节点的个数，
+ subsample：子模型数据的采样比例
+ colsample_bytree：子模型特征的采样比例
+ max_depth：树深度
+ n_estimators：树的个数

# 模型路径
`model/output/model_name_191130_1706.bin`

# 如何预测
```python
import joblib

model=joblib.load('model_path')
# 预测结果，用于分类与回归
pred = model.predict('x_test')
# 预测结果的概率值，仅适用于分类
prob = model.predict_proba('x_test')
```

