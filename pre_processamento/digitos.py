import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
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

#X_treino_normalizado = preprocessing.scale(X_treino)
#scaler = preprocessing.StandardScaler().fit(X_treino)
X_treino = preprocessing.StandardScaler().fit_transform(X_treino)

clf = Perceptron()
rfecv = RFECV(estimator=clf,step=1,cv=StratifiedKFold(2),scoring='accuracy')

rfecv.fit(X_treino,Y_treino)

print("Optimal number of features : {0}".format(rfecv.n_features_))
plt.figure()
plt.xlabel("Número de caracteristicas selecionadas.")
plt.ylabel("Pontuação da Validação Cruzada")
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()
#clf.fit(X_treino,Y_treino)

#X_teste = preprocessing.scale(X_teste)

X_teste = preprocessing.StandardScaler().fit_transform(X_teste)

predicao = rfecv.predict(X_teste)

score = rfecv.score(X_teste,Y_teste)
print(score*100)

confusao = metrics.confusion_matrix(Y_teste,predicao)

for item in confusao:
    print(item)