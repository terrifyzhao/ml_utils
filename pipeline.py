import pandas as pd
from model.model_pipeline import train
import argparse


def read_data(data_path, label_name):
    df = pd.read_csv(data_path)
    X = df.drop([label_name], axis=1).values
    y = df[label_name].values
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("-d", "--data", help="train data", dest="data")
    parser.add_argument("-l", "--label", help="label", dest="label")

    args = parser.parse_args()
    X, y = read_data(args.data, args.label)
    train(X, y)