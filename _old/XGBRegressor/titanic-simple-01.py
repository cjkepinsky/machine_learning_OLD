import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
X = pd.read_csv('../input/train.csv')
X_test_full = pd.read_csv('../input/test.csv')

# print(X.columns.values)

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Survived'], inplace=True)
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Define the model
my_model_1 = XGBRegressor(n_estimators=2400, learning_rate=0.005, max_depth=9, num_parallel_tree=9)
# Fit the model
my_model_1.fit(X_train, y_train)
# my_model_1.fit(X_train, y_train, early_stopping_rounds=1, eval_set=[(X_valid, y_valid)],
#                verbose=False)

# Get predictions
predictions_1 = my_model_1.predict(X_valid)
mae_1 = mean_absolute_error(predictions_1, y_valid)
print("Mean Absolute Error:", mae_1)
# print("pred: ", predictions_1)
# Mean Absolute Error: 0.21371715093728555
