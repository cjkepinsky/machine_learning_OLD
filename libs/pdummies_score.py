import pandas as pd
from sklearn.model_selection import train_test_split


def remove_empty_rows(data, columnNames):
    data.dropna(axis=0, subset=columnNames, inplace=True)
    return data


def remove_columns(data, columnNames):
    data.drop(columnNames, axis=1)
    return data


def separate_target(data, colName):
    y = data[colName]
    X = data.drop([colName], axis=1)
    return X, y


def split_train_test(x_train, y_train, train_size=0.8, test_size=0.2):
    return train_test_split(x_train, y_train, train_size=train_size, test_size=test_size,
                            random_state=0)


def categorical_numerical(x):
    # Select Categorical & numerical columns
    categorical = [cname for cname in x.columns if x[cname].nunique() < 10 and
                   x[cname].dtype == "object"]
    numerical = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]

    return categorical, numerical


def categorical_numerical_cols(X_train_full, X_valid_full, X_test_full):
    # replace categorical data with numerical
    X_train = pd.get_dummies(X_train_full)
    X_valid = pd.get_dummies(X_valid_full)
    X_test = pd.get_dummies(X_test_full)

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_valid, X_test


def grid_stats(grid_model):
    print('Best params:', grid_model.best_params_)
    print('Best score:', grid_model.best_score_)
    print('Mean fit time:', grid_model.cv_results_['mean_fit_time'].mean())
    print()

