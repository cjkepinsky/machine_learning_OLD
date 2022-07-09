import pandas as pd
from cross_validation_score import get_scores
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# print("All: ", data.columns.values)

# Separate target from predictors
data.dropna(axis=0, subset=['Survived'], inplace=True)
y = data.Survived
X = data.drop(['Survived'], axis=1)

# Select Categorical & numerical columns
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and
                    X[cname].dtype == "object"]
# print("Categorical: ", categorical_cols)
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
# print("Numerical: ", numerical_cols)
# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train_full = X[my_cols].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


results = {}
for i in range(1000, 1500, 100):
    model = RandomForestRegressor(n_estimators=i, random_state=0)
    results[i] = get_scores(model, X, y, 5)
    print("Iteration & score: ", i, results[i])

# 1300 is optimal: 0.2946
plt.plot(list(results.keys()), list(results.values()))
plt.show()
