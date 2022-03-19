import pandas as pd
import pdummies_score as pds
import simpleplotter as plot
from sklearn.tree import DecisionTreeClassifier

results = {}
# Read the data
X = pd.read_csv('../input/train.csv')
X_test_full = pd.read_csv('../input/test.csv')

# clean data
# Ticket - 681 unikalnych, prawie jak Name
# Cabin - 204 rekordy, duże braki
# , 'Parch', 'Embarked' - wg feature importance - można usunąć, ale usunięcie powoduje pogorszenie wyników
# 'PassengerId', powoduje przetrenowanie (train data: 1.000 val data: 0.839)
X.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# usuwamy rekordy z brakującymi danymi
X.dropna(axis=0, subset=X.columns, inplace=True)
# print(X.head())
# X_test_full.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
# X_test_full.dropna(axis=0, subset=X_test_full.columns, inplace=True)
# print(X_test_full.head())

# separate target from predictors
X, y = pds.separate_target(X, 'Survived')
# X.info()

# Splitting into train and validation data 80 / 20
X_train_full, X_valid_full, y_train, y_valid = pds.split_train_test(X, y)

X_train, X_valid, X_test = pds.categorical_numerical_cols(X_train_full, X_valid_full, X_test_full)
X_test.dropna(axis=0, subset=X_test.columns, inplace=True)

#
# for md in range(5, 15, 1):
#     model = DecisionTreeClassifier(random_state=0, max_depth=md)
#     model.fit(X_train, y_train)
#     print('md: {:.0f}'.format(md)
#           + ' => train data: {:.3f}'.format(model.score(X_train, y_train))
#           + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))
#     # plot.feature_importance(model, X_train)


model = DecisionTreeClassifier(random_state=0, max_depth=13)
model.fit(X_train, y_train)
print('train data: {:.3f}'.format(model.score(X_train, y_train))
      + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))
# train data: 0.986 val data: 0.783 wg wykresu, feature importance: Embarked, Parch moznaby wywalić,
# niemniej usunięcie tych kolumn powoduje pogorszenie wyników: train data: 0.988 val data: 0.755

plot.feature_importance(model, X_train)
plot.decision_tree(model, X_train)
# na drzewie decyzji widać że algorytm bazuje głównie na PassengerId, co jest bezcelowe, dlatego wywalam.
# train data: 0.965 val data: 0.783

# print(model.predict(X_test)) # jak zweryfikować te prognozy?
