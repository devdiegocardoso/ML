from sklearn import datasets, metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np

iris = datasets.load_iris()

X = iris.data
Y = iris.target

np.random.seed(0)

n_samples = len(Y)
percentage = 0.7

order = np.random.permutation(n_samples)

X = X[order]
Y = Y[order]

Y_teste = Y[int(percentage*n_samples):]
X_teste = X[int(percentage*n_samples):]

Y_treino = Y[:int(percentage*n_samples)]
X_treino = X[:int(percentage*n_samples)]

clf = GaussianNB()

clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

score = clf.score(X_teste,Y_teste)
print(score*100)

confusao = metrics.confusion_matrix(Y_teste,predicao)

for item in confusao:
    print(item)

print(clf.class_count_)
print(clf.class_prior_)
print(clf.sigma_)