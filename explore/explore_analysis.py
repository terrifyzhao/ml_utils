from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import prince
from utils.util import mad_based_outlier
import pandas.api.types as ptypes


def explore_global_plot(data, label='label', n_feats=50, id=None, task='classification'):
    '''
    :param data: DataFrame
    :param label: label column name in the data
    :param n_feats: the number of features be used to analysis.
    :param task: regression or classification
    :return:
    '''
    columns = data.columns.tolist()
    columns.remove(label)

    if id is not None:
        if columns[id].duplicated().sum():
            print('{} is duplicated !!!'.format(id))

        columns.remove(id)
        data.drop(id, axis=1, inplace=True)

    discrete_features = [True if ptypes.is_string_dtype(i) else False for i in data[columns].dtypes]
    discrete_names = [columns[i] for i, v in enumerate(discrete_features) if v]
    numeric_names = list(set(columns) - set(discrete_names))

    if task == 'classification':
        if len(discrete_names):
            # data distribution for each class
            new_data = data.dropna(axis=0)
            famd = prince.FAMD(
                n_components=2,
                n_iter=3,
                copy=True,
                check_input=True,
                engine='auto',
                random_state=42
            )
            famd = famd.fit(new_data[columns])
            ax = famd.plot_row_coordinates(
                new_data,
                ax=None,
                x_component=0,
                y_component=1,
                labels=new_data.index,
                color_labels=['{}'.format(t) for t in new_data[label]],
                ellipse_outline=False,
                ellipse_fill=True,
                show_points=True
            )
            plt.show()
        else:
            new_data = data.dropna(axis=0)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(new_data[columns])
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=label, data=new_data)
            plt.show()

    # correlation plot for all features
    sns.heatmap(data[[label] + columns[:n_feats]].corr())
    plt.show()

    # outlier detection just for numeric features
    outlier = data[numeric_names].apply(mad_based_outlier)
    for i, column in enumerate(outlier.columns):
        print('outlier:\n {}'.format(data[[column]][outlier.iloc[:, i]]))

    # missing value pattern plot for all features
    msno.matrix(data[columns[:n_feats]])
    plt.show()

    msno.bar(data[columns[:n_feats]])
    plt.show()

    miss_data = data[columns[:n_feats]].isnull().sum(axis=1)
    miss_data = miss_data.to_frame()
    miss_data.columns = ['number_of_missing_attributes']
    miss_data.sort_values('number_of_missing_attributes', inplace=True)
    miss_data['index'] = list(range(0, miss_data.shape[0]))
    sns.jointplot(x="index", y="number_of_missing_attributes", data=miss_data)
    plt.show()


def explore_local_plot(data, column):
    '''
    :param data: DataFrame
    :param column: str - the column to be analysis
    :return:
    '''

    # bar plot for distribution
    sns.barplot(data=data[column])
    plt.show()

    # box plot for outlier check
    sns.boxplot(data=data[column])
    plt.show()
