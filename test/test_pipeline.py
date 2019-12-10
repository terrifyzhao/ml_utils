import pandas as pd
from config.base_config import column_config_path, data_path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from preprocess.Preprocess import PreprocessMissvalue, PreprocessOutlier
from feature.FeatureProcess import FeaturesStandard, FeaturesEncoder, FeaturesDecomposition, FeaturesSelection

data = pd.read_csv(data_path)

# simple process
label = 'Survived'
cat_cols = ['Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
data[cat_cols] = data[cat_cols].astype('category')

# read column config
column_config = pd.read_csv(column_config_path)
modules = column_config.columns.tolist()[2:]
columns = column_config.features.tolist()

# default setting
# default for missvalue
column_config.loc[(column_config.types=='category') & (column_config.missvalue.isna()),'missvalue'] ="SimpleImputer(strategy='most_frequent')"
column_config.loc[(column_config.types=='numeric') & (column_config.missvalue.isna()),'missvalue'] = "SimpleImputer(strategy='median')"
# default for encoder
column_config.loc[(column_config.types=='category') & (column_config.encoder.isna()),'encoder'] = "OneHotEncoder"
column_config.fillna('NA', inplace=True)


# construct transformers
column_g = column_config.groupby(modules)
transformer_list = []
for index, each in enumerate(column_g):
    sub_df = each[1]
    sub_tran = each[0]

    sub_feat = sub_df['features'].tolist()

    trans = []
    for i,v in enumerate(sub_tran):
        module = modules[i]
        if module == 'outlier':
            func=PreprocessOutlier
        elif module == 'missvalue':
            func = PreprocessMissvalue
        elif module == 'standard':
            func = FeaturesStandard
        elif module == 'encoder':
            func = FeaturesEncoder

        if v !='NA':
            trans.append((module, func(v)))

    sub_pipe = Pipeline(trans)

    transformer_list.append(('group_'+str(index), sub_pipe, sub_feat))

processes = ColumnTransformer(transformer_list)

dataset = processes.fit_transform(data)
