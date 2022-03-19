import pandas as pd
import pdummies_score as pds
from sklearn.neighbors import KNeighborsClassifier

results = {}
# Read the data
X = pd.read_csv('input/train.csv')
X_test_full = pd.read_csv('input/test.csv')

X.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# usuwamy rekordy z brakującymi danymi
X.dropna(axis=0, subset=X.columns, inplace=True)
# print(X.head())

# separate target from predictors
X, y = pds.separate_target(X, 'Survived')
# X.info()

# Splitting into train and validation data 80 / 20
X_train_full, X_valid_full, y_train, y_valid = pds.split_train_test(X, y, 0.9, 0.1)

X_train, X_valid, X_test = pds.categorical_numerical_cols(X_train_full, X_valid_full, X_test_full)
X_test.dropna(axis=0, subset=X_test.columns, inplace=True)

# for nb in range(1, 50, 1):
#     model = KNeighborsClassifier(n_neighbors=nb, n_jobs=4)
#     model.fit(X_train, y_train)
#     print('nb: {:.0f}'.format(nb)
#           + ' => train data: {:.3f}'.format(model.score(X_train, y_train))
#           + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))
# #     # plot.feature_importance(model, X_train)

model = KNeighborsClassifier(n_neighbors=20)
model.fit(X_train, y_train)
print('train data: {:.3f}'.format(model.score(X_train, y_train))
      + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))
# nb: 20 => train data: 0.727 val data: 0.722


# plot.feature_importance(model, X_train)
# plot.decision_tree(model, X_train)
# na drzewie decyzji widać że algorytm bazuje głównie na PassengerId, co jest bezcelowe, dlatego wywalam.
# train data: 0.965 val data: 0.783

# print(model.predict(X_test)) # jak zweryfikować te prognozy?
