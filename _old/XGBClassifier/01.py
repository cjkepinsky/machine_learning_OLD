import pandas as pd
import pdummies_score as pds
import simpleplotter as plot
from xgboost import XGBClassifier

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

# num_parallel_tree = 10
#
# for i in range(50, 200, 50):
#     for lr in np.arange(0.06, 0.1, 0.01):
#         for md in range(5, 9, 1):
#             model = XGBClassifier(n_estimators=i, learning_rate=lr, max_depth=md, num_parallel_tree=num_parallel_tree,
#                                   use_label_encoder=False, eval_metric='error')
#             model.fit(X_train, y_train)
#             train_score = model.score(X_train, y_train)
#             valid_score = model.score(X_valid, y_valid)
#             if valid_score > max_valid:
#                 max_train = train_score
#                 max_valid = valid_score
#                 max_est = i
#                 max_lr = lr
#                 max_md = md
#
#             # if valid_score > 0.8:
#             print('train_perc: {:.2f}'.format(train_perc) + ' valid_perc: {:.2f}'.format(
#                 valid_perc) + ' est: {:.0f}'.format(i) + ' lr: {:.3f}'.format(lr) + ' md: {:.0f}'.format(md)
#                   + ' => train data: {:.3f}'.format(train_score)
#                   + ' val data: {:.3f}'.format(valid_score))

#             # plot.feature_importance(model, X_train)
#
# print('# -- num_parallel_tree: {:.0f}'.format(num_parallel_tree) + ' train_perc: {:.2f}'.format(
#     train_perc) + ' valid_perc: {:.2f}'.format(valid_perc) + ' est: {:.0f}'.format(max_est) + ' lr: {:.3f}'.format(
#     max_lr) + ' md: {:.0f}'.format(max_md)
#       + ' => train data: {:.3f}'.format(max_train)
#       + ' val data: {:.3f}'.format(max_valid))


# num_parallel_tree: 10 train_perc: 0.80 valid_perc: 0.20 est: 150 lr: 0.080 md: 7 => train data: 0.951 val data: 0.853


model = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=7, num_parallel_tree=10,
                      use_label_encoder=False, eval_metric='error')
model.fit(X_train, y_train)
print('train data: {:.3f}'.format(model.score(X_train, y_train))
      + ' val data: {:.3f}'.format(model.score(X_valid, y_valid)))

plot.feature_importance(model, X_train)
plot.feature_importance(model, X_valid)

