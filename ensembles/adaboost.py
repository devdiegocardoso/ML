from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
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

#clf = KNeighborsClassifier(n_neighbors=5)

clf = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(clf,X_treino,Y_treino,cv=5)
print(scores.mean())

#clf = BaggingClassifier(SVC(),n_estimators=10,max_samples=0.5,max_features=0.5)

clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

score = clf.score(X_teste,Y_teste)
print(score*100)

confusao = metrics.confusion_matrix(Y_teste,predicao)

for item in confusao:
    print(item)