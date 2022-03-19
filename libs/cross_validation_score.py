import statistics as stat

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def separate_target(data, colName):
    # Separate target from predictors
    data.dropna(axis=0, subset=[colName], inplace=True)
    y = data[colName]
    X = data.drop([colName], axis=1)
    return X, y


def split_columns(x):
    # Select Categorical & numerical columns
    categorical = [cname for cname in x.columns if x[cname].nunique() < 10 and
                   x[cname].dtype == "object"]
    numerical = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]

    return categorical, numerical


def split_train_test_80(x_train, y_train):
    return train_test_split(x_train, y_train, train_size=0.8, test_size=0.2,
                            random_state=0)


def preprocess(X):
    categorical_cols, numerical_cols = split_columns(X)
    my_cols = categorical_cols + numerical_cols
    numerical_transformer = SimpleImputer(strategy='constant')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    X_train_full = X[my_cols].copy()

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]), X_train_full


def get_scores(model, preprocessor, X_train, y_train, cv):
    # - Pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', model)
                                     ])

    scores = -1 * cross_val_score(model_pipeline, X_train, y_train,
                                  cv=cv,
                                  scoring='neg_mean_absolute_error')

    return stat.median_grouped(scores, interval=1)
