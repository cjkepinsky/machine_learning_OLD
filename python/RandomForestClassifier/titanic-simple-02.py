import pandas as pd
from sklearn.metrics import mean_absolute_error
import pdummies_score as pds
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# https://www.kaggle.com/alexisbcook/titanic-tutorial

results = {}
# Read the data
X = pd.read_csv('../input/train.csv')
X_test_full = pd.read_csv('../input/test.csv')

# Remove rows with missing target, separate target from predictors
X, y = pds.separate_target(X, 'Survived')

# Splitting and preprocessing
X_train_full, X_valid_full, y_train, y_valid = pds.split_train_test(X, y)
X_train, X_valid, X_test = pds.categorical_numerical_cols(X_train_full, X_valid_full, X_test_full)
print(X_train.head())
# jakie wymiary
# ile parametrów
# które kolumny wywalić, okroić zbiór do 4-5 kolumn
# usunąć puste rekordy dla każdej kolumny lub uzupełnić avg
# ostateczny kod przekleić do jupytera
# https://mljar.com/blog/save-load-random-forest/

for i in range(2200, 2500, 100):
    model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)
    # model = XGBRegressor(n_estimators=i, learning_rate=0.005, max_depth=9, num_parallel_tree=9)
    model.fit(X_train, y_train)
    # my_model_1.fit(X_train, y_train, early_stopping_rounds=1, eval_set=[(X_valid, y_valid)], verbose=False)
    predictions = model.predict(X_valid)
    print(predictions[0:10])
    results[i] = mean_absolute_error(predictions, y_valid)
    print(y_valid[0:10])
    print("n_estimators, mae:", i, results[i])

# n_estimators, mae: 2300 0.2133573348593785
# Regressor - przewidywanie przyszłych wartości, zakres nieznany
# Classifier - wartości znane, znany zbiór wartości
# XGBoost Classifier


# for i in range(1300, 1600, 100):
#     model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)
#     # model = RandomForestRegressor(n_estimators=500, random_state=0)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_valid)
#     results = mean_absolute_error(predictions, y_valid)
#     print("mae:", results)
# ValueError: Input contains NaN, infinity or a value too large for dtype('float32').


plt.plot(list(results.keys()), list(results.values()))
plt.show()
