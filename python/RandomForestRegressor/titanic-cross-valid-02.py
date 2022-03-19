import pandas as pd
import cross_validation_score as crv
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

results = {}
# Read the data
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

X, y_train_full = crv.separate_target(data, "Survived")

preprocessor, X_train_full = crv.preprocess(X)

X_train, X_valid, y_train, y_valid = crv.split_train_test_80(X_train_full, y_train_full)

# print("XGBRegressor:")
# for i in range(2300, 2500, 100):
#     model = XGBRegressor(n_estimators=i, learning_rate=0.005, max_depth=9, num_parallel_tree=9)
#     results[i] = crv.get_scores(model, preprocessor, X_train, y_train, 5)
#     print("Iteration & score: ", i, results[i])

# Iteration & score:  2300 0.26774923296228814


for i in range(1200, 1600, 100):
    model = RandomForestRegressor(n_estimators=i, random_state=0)
    results[i] = crv.get_scores(model, preprocessor, X_train, y_train, 5)
    print("Iteration & score: ", i, results[i])

# Iteration & score:  1400 0.26664335664335664

#
# for i in range(1200, 1600, 100):
#     model = RandomForestRegressor(n_estimators=i, random_state=0)
#     results[i] = crv.get_scores(model, preprocessor, X_valid, y_valid, 5)
#     print("Iteration & score: ", i, results[i])
#
# 1400 is optimal: 0.266

plt.plot(list(results.keys()), list(results.values()))
plt.show()
