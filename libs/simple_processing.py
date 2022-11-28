import re

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def fillna_val_cols(df, cols, val):
    df.update(df[cols].fillna(val, inplace=True))
    df.describe()
    return df


def fillna_mean(df, col):
    df[col].fillna(df[col].mean(), inplace=True)
    return df


def fillna_val(df, col, val):
    df[col].fillna(val, inplace=True)
    return df


def dropna_rows(df, columnNames):
    return df.dropna(axis=0, subset=columnNames, inplace=False)


def remove_columns(data, columnNames):
    return data.drop(columnNames, axis='columns')


def separate_target(data, column_name):
    y = data[column_name]
    X = data.drop([column_name], axis=1)
    return X, y


def split_train_test(x_train, y_train, train_size=0.8, test_size=0.2, random_state=0):
    return train_test_split(x_train, y_train, train_size=train_size, test_size=test_size,
                            random_state=random_state)


def categorical_numerical(x):
    # Select Categorical & numerical columns
    categorical = [cname for cname in x.columns if x[cname].nunique() < 10 and
                   x[cname].dtype == "object"]
    numerical = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]

    return categorical, numerical


def categorize(train, valid):
    train = pd.get_dummies(train)
    valid = pd.get_dummies(valid)
    return train.align(valid, join='left', axis=1, fill_value=0)


def categorize_train_valid_test(X_train_full, X_valid_full, X_test_full):
    # replace categorical data with numerical
    X_train = pd.get_dummies(X_train_full)
    X_valid = pd.get_dummies(X_valid_full)
    X_test = pd.get_dummies(X_test_full)

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_valid, X_test


def print_scores(y_valid, y_pred):
    print('Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
    print('Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False), '/', y_pred.shape[0])
    print('Precision score: ', precision_score(y_valid, y_pred, average='weighted'))
    print('Recall score: ', recall_score(y_valid, y_pred, average='weighted'))
    print('F1 score: ', f1_score(y_valid, y_pred, average='weighted'))
    print()

#
# @dispatch(GridSearchCV, DataFrame, Series)
# def print_scores(grid_model, X_valid, y_valid):
#     y_pred = grid_model.predict(X_valid)
#     print(grid_model.best_params_)
#     print('Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
#     print('Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False), '/', y_pred.shape[0])
#     print('Precision score: ', precision_score(y_valid, y_pred, average='weighted'))
#     print('Recall score: ', recall_score(y_valid, y_pred, average='weighted'))
#     print('F1 score: ', f1_score(y_valid, y_pred, average='weighted'))
#     print()


# helper function to retrieve model name from model object
def get_model_name(trained_model_obj):
    reg = re.compile('([A-Za-z]+)\(')
    return reg.findall(trained_model_obj.__str__())[0]


# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
def normalize(df):
    cols = df.columns
    scaler = MinMaxScaler()

    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=cols)

    return df
