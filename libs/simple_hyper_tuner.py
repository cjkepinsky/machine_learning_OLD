# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
import sys

from libs.simple_processing import get_model_name, categorize, categorize_train_valid_test
from libs.simpleplotter import simple_roc
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# LinearRegression (LR)
# RandomForest (RF)
# Support Vector Machine (SVM)


def quick_gridsearchcv_overview(X, y, cv=None, train_size=None, random_state=None, verbose=None):

    if cv is None:
        cv = [2]
    if random_state is None:
        random_state = [0]
    if train_size is None:
        train_size = [0.8]
    if verbose is None:
        verbose = 1

    params = [
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': GaussianNB(),
            'hyperparams': {}
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': KNeighborsClassifier(
                # n_neighbors=19
            ),
            'hyperparams': {
                'n_neighbors': range(10, 25, 5)
            }
        },
        # {
        #     'splitter': {'train_size': train_size, 'random_state': random_state},
        #     'cv': cv,
        #     'model': LinearRegression(),
        #     'hyperparams': {}
        # },
        {   # Binary or Multiclass as One-Vs-One
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': svm.LinearSVC(max_iter=4000),
            'hyperparams': {}
        },
        {   # Binary or Multiclass - One-Vs-One and One-vs-Rest
            # https://www.baeldung.com/cs/svm-multiclass-classification
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': svm.SVC(),
            'hyperparams': {
                'decision_function_shape': ['ovr', 'ovo']
                , 'kernel': ['rbf', 'poly']
                # , 'shrinking': [True, False]
                # , 'coef0': [0, 0.2, 0.4]
                # , 'gamma' : ['scale', 'auto']
                # , 'degree': [1, 2, 3, 4, 5]
                # , 'C': [1, 0.1, 0.4, 0.7, 1.2]
            }
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': LogisticRegression(max_iter=200, multi_class="ovr"),
            'hyperparams': {
                # 'tol': np.arange(0, 0.005, 0.001)
                # , 'C': [1, 0.1, 0.4, 0.7, 1.2]

            }
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': DecisionTreeClassifier(random_state=0),
            'hyperparams': {
                # 'max_depth': [1, 2, 3, 4, 5],
                # 'max_leaf_nodes': range(2, 7, 1)
            }
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': RandomForestClassifier(
                # max_depth=6, max_features="auto", bootstrap=True, oob_score=True, random_state=0
                # , criterion='entropy', n_estimators=75
            ),
            'hyperparams': {
                #         'max_features':["auto", "sqrt", "log2"],
                #         'max_samples': range(50, 80, 5),
                #         'n_estimators': range(40, 80, 10),
                #         'max_depth': range(4, 8, 1)
            }
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': XGBClassifier(
                # n_estimators=70, learning_rate=0.08, max_depth=8, use_label_encoder=True
                # , num_parallel_tree=10
                # , min_samples_leaf=1, min_samples_split=3
                #   use_label_encoder=False, eval_metric='error'
                # , max_leaf_nodes=1, validation_fraction=0
                # , tol=0, ccp_alpha=0, n_iter_no_change=1
            ),
            'hyperparams': {
                # 'n_estimators':range(100, 160, 10)
                # 'min_samples_split':np.arange(3, 6, 1),
                # 'min_samples_leaf':np.arange(1, 4, 1),
                # 'subsample':np.arange(0.6, 0.9, 0.1),
                # 'criterion': ['friedman_mse', 'squared_error'],
                # , 'max_depth': range(5, 9, 1)
                # , 'learning_rate':np.arange(0.09, 0.13, 0.01)
                # , 'min_weight_fraction_leaf':np.arange(0, 0.14, 0.02)
                # , 'min_impurity_decrease':np.arange(0, 0.14, 0.02)
                # 'max_leaf_nodes':range(1, 7, 1)
                # , 'warm_start': [True, False]
                # 'validation_fraction': np.arange(0, 0.005, 0.001)
                # 'tol': np.arange(0, 0.005, 0.001)
                # 'ccp_alpha': np.arange(0, 0.005, 0.001)
                # 'n_iter_no_change':range(1, 7, 1)
            }
        },
        {
            'splitter': {'train_size': train_size, 'random_state': random_state},
            'cv': cv,
            'model': GradientBoostingClassifier(
                # loss='deviance', max_features="auto"
                # , criterion='squared_error', learning_rate=1.2, max_depth=6
                # , n_estimators=75, random_state=6
            ),
            'hyperparams': {
                #             'criterion': ['friedman_mse', 'mse', 'mae'],
                #             # 'loss':['deviance', 'exponential'],
                #             'random_state': range(5, 8, 1),
                #             'n_estimators': range(60, 90, 5),
                #             'learning_rate': np.arange(1, 1.4, 0.1),
                #             'max_depth': range(4, 9, 1)
            }
        }
    ]

    return gridsearchcv_tuner(X, y, params, verbose)


def gridsearchcv_tuner(X, y, params, verbose=1, do_categorize=True, X_test=None):
    best_model = {}
    best_f1 = 0

    print("X shape: ", X.shape)

    for p in params:
        for train_size in p['splitter']['train_size']:
            for random_state in p['splitter']['random_state']:
                if not p['reducer']:
                    print('> Reducer: none')
                    X_reduced = X
                else:
                    reducer = p['reducer']
                    print('> Reducer:', reducer)
                    reducer_model = reducer.fit(X)
                    X_reduced = reducer_model.transform(X)
                    # X_reduced = reducer_model.embedding_
                    print('- X reduced shape: ', X_reduced.shape)
                    # plt.scatter(reducer_model.embedding_[:, 0], reducer_model.embedding_[:, 1], s=5, c=y, cmap='Spectral')
                    # plt.title('Embedding of the training set by reducer', fontsize=24)

                X_train, X_valid, y_train, y_valid = train_test_split(X_reduced, y, train_size=train_size, random_state=random_state)
                if do_categorize is True:
                    if X_test is None:
                        X_train, X_valid = categorize(X_train, X_valid)
                    else:
                        X_train, X_valid, X_test = categorize_train_valid_test(X_train, X_valid, X_test)

                for cv in p['cv']:
                    print('> Model:', get_model_name(p['model']))

                    grid_model = GridSearchCV(p['model'], p['hyperparams'], cv=cv, n_jobs=None, verbose=0)
                    grid_model.fit(X_train, y_train)

                    f1 = predict_print_valid_scores(grid_model, X_valid, y_valid, verbose)

                    if verbose == 0:
                        print("- cv: ", cv)
                    if verbose > 0:
                        print("GridSearchCV Training Results:")
                        print("- Best Score: ", grid_model.best_score_)
                        print("Params:")
                        print('- cv: ', cv)
                        print("- Splitter Params: ", {'train_size': train_size, 'random_state': random_state})
                        print("- Model Params: ", p['model'])
                        print("- Best H-Params: ", grid_model.best_params_)
                    if verbose == 2:
                        print("- All Params: ")
                        print(grid_model.get_params())
                    print()
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = grid_model.best_estimator_

    print('Done.')
    return best_model


def predict_print_valid_scores(model, X_valid, y_valid, verbose=1):
    y_pred = model.predict(X_valid)
    if verbose > 0:
        print('- Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
        print('- Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False), '/', y_pred.shape[0])
        print('- Precision score: ', precision_score(y_valid, y_pred, average='weighted'))
        print('- Recall score: ', recall_score(y_valid, y_pred, average='weighted'))
    f1 = f1_score(y_valid, y_pred, average='weighted')
    print('- F1 score: ', f1)
    simple_roc(y_valid, y_pred)
    return f1


def valid_score_tuner(X_train, y_train, X_valid, y_valid, parameters):
    max_train = 0
    max_valid = 0
    max_params = {}
    max_model_name = ''
    max_model = {}
    i = 0
    train_results = []
    test_results = []

    for p in parameters:
        modelName = p['modelName']
        hyperParamNames = p['hyperParamNames']
        hyperParamValues = p['hyperParamValues']
        params = p['params']
        fits = len(hyperParamValues[0]) * len(hyperParamValues[1]) * len(hyperParamValues[2])
        print('Fits to be done: ', fits)
        short_name = modelName.__name__

        for r1 in hyperParamValues[0]:
            for r2 in hyperParamValues[1]:
                for r3 in hyperParamValues[2]:

                    i += 1
                    sys.stdout.write(str(i) + '.. ')

                    hp = {
                        hyperParamNames[0]: r1,
                        hyperParamNames[1]: r2,
                        hyperParamNames[2]: r3
                    }

                    model = modelName(**hp, **params)
                    model.fit(X_train, y_train)

                    train_score = model.score(X_train, y_train)
                    valid_score = model.score(X_valid, y_valid)

                    if valid_score > max_valid:
                        max_model_name = short_name
                        max_model = model
                        max_train = train_score
                        max_valid = valid_score
                        max_params = hp

                        #                 if valid_score >= min_valid_score:
                        print()
                        print(short_name
                              + ' => train data SCORE: {:.3f}'.format(train_score)
                              + ' val data SCORE: {:.3f}'.format(valid_score))
                        print('params: ', max_params)

                        # for binary classifiers only
                        # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
                        # roc_auc = auc(false_positive_rate, true_positive_rate)
                        # train_results.append(roc_auc)
                        #
                        # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_valid, y_pred)
                        # roc_auc = auc(false_positive_rate, true_positive_rate)
                        # test_results.append(roc_auc)

    print()
    print('WINNER: ' + max_model_name)
    print(' => train data score: {:.3f}'.format(max_train)
          + ' val data score: {:.3f}'.format(max_valid))
    print('BEST params: ', max_params)

    y_pred = max_model.predict(X_valid)
    print_valid_scores(y_valid, y_pred)

    return max_model


def print_valid_scores(y_valid, y_pred):
    print('Accuracy score: ', accuracy_score(y_valid, y_pred, normalize=True))
    print('Accuracy count: ', accuracy_score(y_valid, y_pred, normalize=False), '/', y_pred.shape[0])
    print('Precision score: ', precision_score(y_valid, y_pred, average='weighted'))
    print('Recall score: ', recall_score(y_valid, y_pred, average='weighted'))
    print('F1 score: ', f1_score(y_valid, y_pred, average='weighted'))
    print()


#%%
