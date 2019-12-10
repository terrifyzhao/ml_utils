import pandas as pd
from config.base_config import data_path
from explore.explore_analysis import explore_global_plot, explore_local_plot
import warnings

warnings.filterwarnings('ignore')

data=pd.read_csv(data_path)

cat_cols = ['Pclass','Siblings/Spouses Aboard','Parents/Children Aboard']
data[cat_cols] = data[cat_cols].astype('category')
data = data.drop('Name',axis=1)

explore_global_plot(data, label='Survived')

explore_local_plot(data,'Age')