import pandas as pd
import pdummies_score as pds
from sklearn.ensemble import RandomForestClassifier

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
print(X.describe(include=['O']))

# clean data
# Ticket - 681 unikalnych, prawie jak Name
# Cabin - 204 rekordy, duże braki
X.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
# usuwamy rekordy z brakującymi danymi
X.dropna(axis=0, subset=X.columns, inplace=True)

# print(X.head())

# X = pds.remove_columns(X, ['Name', 'PassengerId', 'Ticket', 'Cabin'])
# X = pds.remove_empty_rows(X, X.columns)

# separate target from predictors
X, y = pds.separate_target(X, 'Survived')
# X.info()

# Splitting and preprocessing
X_train_full, X_valid_full, y_train, y_valid = pds.split_train_test(X, y, 0.4, 0.6)
X_train, X_valid, X_test = pds.categorical_numerical_cols(X_train_full, X_valid_full, X_test_full)

# print(X_train.head())
# jakie wymiary
# ile parametrów
# które kolumny wywalić, okroić zbiór do 4-5 kolumn
# usunąć puste rekordy dla każdej kolumny lub uzupełnić avg
# ostateczny kod przekleić do jupytera
# https://mljar.com/blog/save-load-random-forest/
# X_train.info()
# X_valid.info()
# X_test.info()


for i in range(200, 500, 100):
    for md in range(1, 5, 1):
        model = RandomForestClassifier(n_estimators=i, n_jobs=4, max_depth=md, random_state=0)
        model.fit(X_train, y_train)
        print(model.__str__()
              + ' => train data: {:.3f}'.format(model.score(X_train, y_train))
              + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))

# model = RandomForestClassifier(n_estimators=200, n_jobs=4, max_depth=14, random_state=0)
# model.fit(X_train, y_train)
# print('train data: {:.3f}'.format(model.score(X_train, y_train))
#       + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))

# (split 0.4, 0.6) est: 300 md: 3 => train data: 0.845 val data: 0.813
# est: 400 md: 7 => train data: 0.935 val data: 0.812


# plot.feature_importance(model, X_train)


#%%
