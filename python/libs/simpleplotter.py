# https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
import time
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Source
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def feature_importance(model, X):
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.keys())
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.show()


def decision_tree(model, X):
    timestamp = time.monotonic_ns()
    path = 'tree_' + str(timestamp) + '.dot'
    export_graphviz(model, out_file=path, feature_names=X.columns,
                    impurity=False, filled=True)
    s = Source.from_file(path)
    s.render('tree', format='jpg', view=True)
    # https://scikit-learn.org/stable/modules/tree.html


def chart(results_arr):
    plt.plot(list(results_arr.keys()), list(results_arr.values()))
    plt.show()


def simple_roc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC', fontsize=18)
    # more: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    # https://community.ibm.com/community/user/cloudpakfordata/blogs/harris-yang1/2021/05/26/scikit-learn-churn-model-cpd35
    # https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv


def simple_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='BrBG')
    plt.show()

    return cm

