# def print_summary(y_test, y_pred):
#     res = summary(y_test, y_pred)
#     print('Test data count: ', res['data_count'])
#     print('accuracy count: ', res['accuracy_count'])
#     print('accuracy score: ', res['accuracy'])
#     print('precision score: ', res['precision'])
#     print('recall score: ', res['recall'])
#     print()

import re
import sys


# X_train, y_train, X_valid, y_valid, p['modelName'], p['params'], p['hyperParams']
def get_scoring(X_train, y_train, X_valid, y_valid, modelName, hyperParamNames, hyperParamValues, params=None):
    if params is None:
        params = {}
    max_train = 0
    max_valid = 0
    max_model = ''
    max_params = {}
    max_model_name = ''
    i = 0
    train_results = []
    test_results = []

    reg = re.compile('([A-Za-z]+)\(')
    fits = len(hyperParamValues[0]) * len(hyperParamValues[1]) * len(hyperParamValues[2])
    print('Fits to be done: ', fits)
    short_name = reg.findall(modelName.__str__())[0]

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
                # train_pred = model.predict(X_train)

                if valid_score > max_valid:
                    max_model_name = short_name
                    max_model = model
                    max_train = train_score
                    max_valid = valid_score
                    max_params = hp

                    #                 if valid_score >= min_valid_score:
                    print()
                    print(short_name
                          + ' => train data: {:.3f}'.format(train_score)
                          + ' val data: {:.3f}'.format(valid_score))
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
    print(' => train data: {:.3f}'.format(max_train)
          + ' val data: {:.3f}'.format(max_valid))
    print('BEST params: ', max_params)
    # y_pred = max_model.predict(X_valid)
    # df = DataFrame(y_valid, y_pred)
    # print(df)

    # import matplotlib.pyplot as plt
    # from matplotlib.legend_handler import HandlerLine2D
    # line1, = plt.plot(hyperParamValues[1], train_results, 'b', label='Train AUC')
    # line2, = plt.plot(hyperParamValues[1], test_results, 'r', label='Test AUC')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    # plt.ylabel('AUC score')
    # plt.xlabel('learning rate')
    # plt.show()

    # return {model: max_model, max_train: max_train, max_valid: max_valid}
