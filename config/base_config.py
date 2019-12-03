# 随机种子
random_state = 0
# 数据集切分比例
split_size = 0.3
# 评测指标
eval_metric = ['auc', 'acc', 'recall', 'f1']
# 需要使用的模型名字
"""
'logistic_regression' 逻辑回归
'svc' svm 分类 
'forest_cls' 随机森林分类
'gbdt_cls' gbdt分类
'linear_regression', 线性回归
'svr' svm回归
'forest_reg' 随机森林回归 
'gbdt_reg' gbdt回归
"""
# model_name = ['logistic_regression']
model_name = ['logistic_regression', 'svc', 'forest_cls', 'gbdt_cls', 'lightgbm_cls', 'xgboost_cls']
# model_name = ['linear_regression', 'svr', 'forest_reg', 'gbdt_reg','lightgbm_reg','xgboost_reg']
