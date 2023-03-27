from sklearn.ensemble import VotingClassifier
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import numpy as np

iris = datasets.load_digits()

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

#X_treino = StandardScaler().fit_transform(X_treino)
#X_teste = StandardScaler().fit_transform(X_teste)

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = SVC(probability=True)
#clf3 = GaussianNB()
clf3 = MLPClassifier(max_iter=1000,hidden_layer_sizes=(100,))

clf = VotingClassifier(
    estimators=[('knn',clf1),('svm',clf2),('mlp',clf3)],
    voting='soft',weights=[1,2,1]
    )

#clf1 = RFECV(clf1,step=1,cv=StratifiedKFold(5),scoring='accuracy')
#clf2 = RFECV(clf2,step=1,cv=StratifiedKFold(5),scoring='accuracy')
#clf3 = RFECV(clf3,step=1,cv=StratifiedKFold(5),scoring='accuracy')

for clf, label in zip([clf1,clf2,clf3,clf],['KNN','SVM','MLP','Ensemble']):
    scores = cross_val_score(clf,X_treino,Y_treino,scoring='accuracy',cv=5)
    print("Score: {0:.2} (+/- {1:.2}) [{2}]".format(scores.mean(),scores.std(),label))

clf.fit(X_treino,Y_treino)

predicao = clf.predict(X_teste)

score = clf.score(X_teste,Y_teste)
print(score*100)

confusao = metrics.confusion_matrix(Y_teste,predicao)

for item in confusao:
    print(item)