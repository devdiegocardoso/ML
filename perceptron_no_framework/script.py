def HL(N):
    return 1 if N >= 0 else 0

def predict(v1,v2,bias):
    acc = 0
    for i in range(0,len(v1)):
        acc += v1[i] * v2[i]
    return acc + bias

X = [[2,2],[-2,-2],[-2,2],[-1,1]]
Y = [0,1,0,1]

w = [0,0]
b = 0

ciclos = -1
epocas = 0
while ciclos != 0:
    ciclos = 0
    epocas += 1
    for i in range(0,len(X)):
        N = HL(predict(X[i],w,b))
        e = Y[i] - N
        if e != 0:
            for j in range(0,len(w)):
                w[j] = w[j] + (X[i][j] * e)
            b = b + e
            ciclos += 1
        
print('Total de Épocas:',epocas)
print('Pesos:',w)
print('bias',b)

w = [2,1]
b = 1

for i in range(0,len(X)):
    P = HL(predict(X[i],w,b))
    print('Predição:',P)