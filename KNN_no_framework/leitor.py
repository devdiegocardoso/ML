#Muda o rótulo de uma classe para inteiro
def replace_label(classe):
    if 'Iris-setosa' in classe:
        return 0
    elif 'Iris-versicolor' in classe:
        return 1
    elif 'Iris-virginica' in classe:
        return 2

def le_matriz(file):
    #Lê o arquivo e armazena em um vetor. Cada elemento do Vetor é uma linha.
    linhas = file.readlines()

    matriz = []

    for linha in linhas:
        #Separa os dados em uma lista
        vetor = linha.split(',')
        for i in range(0,4):
            #Converte os 4 primeiros elementos em float
            vetor[i] = float(vetor[i])
        #Muda o elemento 5 da matriz para um inteiro, que representa um rótulo
        vetor[4] = replace_label(vetor[4])
        #Adiciona a linha formatada na matriz
        matriz.append(vetor)
    file.close()
    return matriz


