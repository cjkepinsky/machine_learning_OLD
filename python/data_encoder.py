import pandas as pd


def pandas_encoder(x_train, x_valid):
    # One-hot encode the data (to shorten the code, we use pandas)
    x_train = pd.get_dummies(x_train)
    x_valid = pd.get_dummies(x_valid)
    # x_test = pd.get_dummies(x_test)
    x_train, x_valid = x_train.align(x_valid, join='left', axis=1)
    # x_train, x_test = x_train.align(x_test, join='left', axis=1)
    return x_train, x_valid
