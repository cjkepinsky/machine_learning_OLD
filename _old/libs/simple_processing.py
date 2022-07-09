from sklearn.model_selection import train_test_split
import pandas as pd


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


def print_scores(y_valid, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print('Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
    print('Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False), '/', y_pred.shape[0])
    print('Precision score: ', precision_score(y_valid, y_pred))
    print('Recall score: ', recall_score(y_valid, y_pred))
    print('F1 score: ', f1_score(y_valid, y_pred))


# helper function to retrieve model name from model object
def get_model_name(trained_model_obj):
    reg = re.compile('([A-Za-z]+)\(')
    return reg.findall(trained_model_obj.__str__())[0]
