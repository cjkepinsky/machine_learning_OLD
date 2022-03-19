import matplotlib.pyplot as plt
import mglearn.plots

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
#plt.show()


mglearn.plots.plot_knn_classification(n_neighbors=1)