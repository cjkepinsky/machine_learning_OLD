import re
import pandas as pd
import libs.pdummies_score as pds
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# https://www.kaggle.com/alexisbcook/titanic-tutorial
# https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook
# jupyter-lab


results = {}
# Read the data
X = pd.read_csv('input/train.csv')
X_test_full = pd.read_csv('input/test.csv')



# overview of column types and missing data
# X.describe()
# print(X.describe(include='object'))
# print(X.describe(include='float'))
X.info()

# women = X.loc[X.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# print("% of women who survived:", rate_women)

# czemu to nie działą w pycharm
# grid = sns.FacetGrid(X, row='Pclass', col='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# print(grid)
# pl = sns.distplot(x=X.Age, hist=True)

# Process Name column to extract surnames and see if having categorical Surnames would make results better
def get_surname(name):
    reg = re.compile('([A-Za-z]+),')
    return reg.findall(name)[0]


print(X.shape)
X['Surname'] = X.apply(lambda row: get_surname(row['Name']), axis='columns')

# g = sns.FacetGrid(X, col='Survived')
# g.map(plt.hist, 'Surname', bins=5)
# sns.heatmap(X.isnull(), cbar=False)
# X["Age"].isnull().fill(meanAge)

# print(pd.crosstab(X['Surname'], X['Survived']))

# clean data
# Ticket - 681 unikalnych, prawie jak Name
# Cabin - 204 rekordy, duże braki
# , 'Parch', 'Embarked' - wg feature importance - można usunąć, ale usunięcie powoduje pogorszenie wyników
# 'PassengerId', powoduje przetrenowanie (train data: 1.000 val data: 0.839)
X.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis='columns', inplace=True)
# X.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# usuwamy rekordy z brakującymi danymi
X.dropna(axis='index', subset=X.columns, inplace=True)
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

print(X_train)
print(X_valid)


for i in range(350, 500, 50):
    for lr in np.arange(0.020, 0.04, 0.005):
        for md in range(4, 7, 1):
            model = GradientBoostingClassifier(n_estimators=i, random_state=0, learning_rate=lr, max_depth=md)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)
            if valid_score > max_valid:
                max_train = train_score
                max_valid = valid_score
                max_est = i
                max_lr = lr
                max_md = md

            if valid_score > 0.8:
                print('train_perc: {:.2f}' .format(train_perc) + ' valid_perc: {:.2f}' .format(valid_perc) + ' est: {:.0f}'.format(i) + ' lr: {:.3f}'.format(lr) + ' md: {:.0f}'.format(md)
                      + ' => train data: {:.3f}'.format(train_score)
                      + ' val data: {:.3f}'.format(valid_score))
#
#             # plot.feature_importance(model, X_train)
#
print('# train_perc: {:.2f}' .format(train_perc) + ' valid_perc: {:.2f}' .format(valid_perc) + ' est: {:.0f}'.format(max_est) + ' lr: {:.3f}'.format(max_lr) + ' md: {:.0f}'.format(max_md)
      + ' => train data: {:.3f}'.format(max_train)
      + ' val data: {:.3f}'.format(max_valid))

# train_perc: 0.80 valid_perc: 0.20 est: 400 lr: 0.030 md: 5 => train data: 0.982 val data: 0.860
# train_perc: 0.80 valid_perc: 0.20 est: 450 lr: 0.030 md: 5 => train data: 0.982 val data: 0.853
# est: 400 lr: 0.030 md: 5 => train data: 0.982 val data: 0.860
# est: 200 lr: 0.030 md: 4 => train data: 0.949 val data: 0.817
# est: 800.000 train data: 0.993 val data: 0.776
# est: 850 lr: 0.050 md: 6 train data: 0.993 val data: 0.846

model = GradientBoostingClassifier(n_estimators=400, random_state=0, learning_rate=0.03, max_depth=5, verbose=1)
model.fit(X_train, y_train)
# print('train data: {:.3f}'.format(model.score(X_train, y_train))
#       + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_pred = model.predict(X_valid)
pred_results = pd.DataFrame({'y_valid' : y_valid,
                             'y_pred': y_pred})


print('Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
# number of correct predictions
print('Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False))
print('Precision score: ', precision_score(y_valid, y_pred))
print('Recall score: ', recall_score(y_valid, y_pred))
print()

# wg wykresu, featury: Embarked, Parch mozna wywalić, niemniej usunięcie tych kolumn powoduje pogorszenie wyników:
# train data: 0.982 val data: 0.860

# plot.feature_importance(model, X_train)
# plot.feature_importance(model, X_valid)


# predictions = model.predict(X_valid)
# print('Predictions: ', predictions[0:10])
# results[i] = mean_absolute_error(predictions, y_valid)
# print('y_Valid: ', y_valid[0:10])
# print("n_estimators, mae:", i, results[i])


# model = GridSearchCV(p['model'], p['hyperparams'], cv=3, return_train_score=True)
