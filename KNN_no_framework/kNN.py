from leitor import *
from operator import itemgetter
#Abre o arquivo
fileTreino = open("treino.data")
fileTeste = open("teste.data")

matriz_treino = le_matriz(fileTreino)
matriz_teste = le_matriz(fileTeste)

k = 5

matriz_confusao = [0] * 3
for i in range(0,3):
    matriz_confusao[i] = [0] * 3

def voto(distancias):
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(0,k):
        if distancias[i][1] == 0:
            c1 += 1
        if distancias[i][1] == 1:
            c2 += 1
        if distancias[i][1] == 2:
            c3 += 1
    if c1 > c2 and c1 > c3:
        return 0
    elif c2 > c1 and c2 > c3:
        return 1
    else:
        return 2

def manhattan(x,y,attr):
    soma = 0
    for i in range(0,attr):
        soma += abs(x[i] - y[i])
    return soma

def adiciona_matriz(x,y):
    matriz_confusao[x][y] += 1

def imprimir_matriz(matriz):
    for item in matriz:
        print(item)

def calcular_taxa(matriz_confusao):
    acertos = 0
    erros = 0
    for i in range(0,3):
        for j in range(0,3):
            if i == j:
                acertos += matriz_confusao[i][j]
            else:
                erros += matriz_confusao[i][j]
    total = acertos + erros
    return (acertos / total) * 100


for x in matriz_teste:
    classe_teste = x[4]
    distancias = []
    for y in matriz_treino:
        classe_treino = y[4]
        distancia = manhattan(x,y,4)
        distancias.append((distancia,classe_treino))
    distancias.sort(key=itemgetter(0))
    classificacao = voto(distancias)
    adiciona_matriz(classe_teste,classificacao)

print("Taxa de acerto: {0:.2f}%".format(calcular_taxa(matriz_confusao)))
imprimir_matriz(matriz_confusao)