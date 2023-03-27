from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
import numpy as np
import matplotlib.pyplot as plt

boston = datasets.load_boston()

X = boston.data
Y = boston.target

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

regr1 = LinearRegression()
regr2 = MLPRegressor(hidden_layer_sizes=(25,15),max_iter=1000)
regr3 = SVR()

ensemble = VotingRegressor(estimators=[('LR',regr1),('MLP',regr2),('SVR',regr3)])

ensemble.fit(X_treino,Y_treino)

predicao = ensemble.predict(X_teste)

R2 = metrics.r2_score(Y_teste,predicao)
MSE = metrics.mean_squared_error(Y_teste,predicao)

print('R2:',R2)
print('MSE:',MSE)
#print(Y_teste)
#print(predicao)

#for i in range(len(Y_teste)):
#    print("Valor Real:",Y_teste[i],"Predição:",predicao[i])

plt.plot(Y_teste,Y_teste,label='Valor Esperado')
plt.scatter(Y_teste,predicao,label='Valor Estimado',color='r')
plt.xlabel('Valores Reais')
plt.ylabel('Predição')
plt.legend()
plt.show()