import pandas as pd
import pdummies_score as pds
import numpy as np
from xgboost import XGBRegressor

# https://www.kaggle.com/alexisbcook/titanic-tutorial
# https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook
# jupyter-lab


results = {}
# Read the data
X = pd.read_csv('input/train.csv')
X_test_full = pd.read_csv('input/test.csv')

# women = X.loc[X.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# print("% of women who survived:", rate_women)

# overview of column types and missing data
# X.describe()
# print(X.describe(include='object'))
# print(X.describe(include='float'))
# X.info()

# clean data
# Ticket - 681 unikalnych, prawie jak Name
# Cabin - 204 rekordy, duże braki
# , 'Parch', 'Embarked' - wg feature importance - można usunąć, ale usunięcie powoduje pogorszenie wyników
# 'PassengerId', powoduje przetrenowanie (train data: 1.000 val data: 0.839)
X.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# usuwamy rekordy z brakującymi danymi
X.dropna(axis=0, subset=X.columns, inplace=True)
# print(X.head())

# separate target from predictors
X, y = pds.separate_target(X, 'Survived')
# X.info()

# Splitting into train and validation data
train_perc = 0.8
valid_perc = 0.2
X_train_full, X_valid_full, y_train, y_valid = pds.split_train_test(X, y, train_perc, valid_perc)

# converting categorical cols into numerical
X_train, X_valid, X_test = pds.categorical_numerical_cols(X_train_full, X_valid_full, X_test_full)
# print(X_train.head())
max_train = 0
max_valid = 0
max_est = 0
max_lr = 0
max_md = 0

for i in range(2300, 2500, 100):
    for lr in np.arange(0.04, 0.07, 0.01):
        for md in range(7, 11, 1):
            # model = XGBRegressor(n_estimators=2400, learning_rate=0.005, max_depth=9, num_parallel_tree=9)
            model = XGBRegressor(n_estimators=i, learning_rate=lr, max_depth=md, num_parallel_tree=10)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)
            if valid_score > max_valid:
                max_train = train_score
                max_valid = valid_score
                max_est = i
                max_lr = lr
                max_md = md

            # if valid_score > 0.8:
            print('train_perc: {:.2f}' .format(train_perc) + ' valid_perc: {:.2f}' .format(valid_perc) + ' est: {:.0f}'.format(i) + ' lr: {:.3f}'.format(lr) + ' md: {:.0f}'.format(md)
                  + ' => train data: {:.3f}'.format(train_score)
                  + ' val data: {:.3f}'.format(valid_score))

#             # plot.feature_importance(model, X_train)
#
# print('# train_perc: {:.2f}' .format(train_perc) + ' valid_perc: {:.2f}' .format(valid_perc) + ' est: {:.0f}'.format(max_est) + ' lr: {:.3f}'.format(max_lr) + ' md: {:.0f}'.format(max_md)
#       + ' => train data: {:.3f}'.format(max_train)
#       + ' val data: {:.3f}'.format(max_valid))


#
# model = GradientBoostingClassifier(n_estimators=400, random_state=0, learning_rate=0.03, max_depth=5)
# model.fit(X_train, y_train)
# print('train data: {:.3f}'.format(model.score(X_train, y_train))
#       + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))

# wg wykresu, featury: Embarked, Parch mozna wywalić, niemniej usunięcie tych kolumn powoduje pogorszenie wyników:
# train data: 0.982 val data: 0.860

# plot.feature_importance(model, X_train)
# plot.feature_importance(model, X_valid)


# predictions = model.predict(X_valid)
# print('Predictions: ', predictions[0:10])
# results[i] = mean_absolute_error(predictions, y_valid)
# print('y_Valid: ', y_valid[0:10])
# print("n_estimators, mae:", i, results[i])
