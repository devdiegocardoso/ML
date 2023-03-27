import math
import numpy as np
import time
import getopt, sys

def manhattan_distance(v1,v2,N):
    sum = 0.0
    for i in range(N):
        sum+= abs(v1[i] - v2[i])
    return sum

def euclidian_distance(v1,v2,N):
    sum = 0.0
    for i in range(N):
        sum+= (v1[i] - v2[i]) * (v1[i] - v2[i])
    return math.sqrt(sum)


def min_max(v1,N):
    xmin = min(v1)
    xmax = max(v1)

    for i in range(N):
        v1[i] = (v1[i] - xmin) / (xmax - xmin)

def z_score(v1,N):
    avgX = np.mean(v1)
    stdevX = np.std(v1)
    print(avgX)
    print(stdevX)
    for i in range(N):
        v1[i] = (v1[i] - avgX) / stdevX

def knn(training_base,test_base,K):
    evaluation = [0] * len(test_base)
    start_time = time.time()
    distances = np.array([])
    distances = np.zeros(len(training_base))
    confusion_matrix = np.matrix([[]])
    confusion_matrix = [[0,0],[0,0]]
    labels = 2
    correct_eval = 0
    wrong_eval = 0
    accuracy = 0
    for i in range(len(test_base)):
    #for i in range(iStart,iEnd):
        print("{0} samples classified.".format(i))
        for j in range(len(training_base)):
            distances[j] = manhattan_distance(test_base[i],training_base[j],len(test_base[i])-1)
        kIndex = distances.argsort()[:K]
        positive = 0
        negative = 0
        for k in kIndex:
            if training_base[k][-1] == 1:
                positive+= 1
            else:
                negative+= 1
        if positive > negative:
            evaluation[i] = 1
        else:
            evaluation[i] = 0
    for c in range(len(test_base)):
        if test_base[c][-1] == evaluation[c]:
            correct_eval+= 1
        else:
            wrong_eval+= 1
        confusion_matrix[int(test_base[c][-1].item())][evaluation[c]]+= 1
    
    accuracy = correct_eval / len(test_base)
    print("Accuracy: ",accuracy)
    f = open("knn_final","w+")
    f.write("Accuracy: %lf\n" % accuracy)
    f.write("Confusion Matrix:\n")
    elapsed_time = time.time() - start_time
    for i in range(labels):
        f.write(str(confusion_matrix[i]))
        f.write("\n")
    f.write("Total time: %lf\n" % elapsed_time)
    print("Total time: \n",elapsed_time)
    f.close()

def main():
    trainBaseFile = ''
    testBaseFile = ''
    K = 0
    opt, args = getopt.getopt(sys.argv[1:],'')
    if(len(args) == 3):
        trainBaseFile = args[0]
        testBaseFile = args[1]
        K = int(args[2])

        training_base = np.load(trainBaseFile)
        test_base = np.load(testBaseFile)

        print("Normalizing Bases.")
        for i in range(len(training_base)):
            min_max(training_base[i],len(training_base[i])-1)

        for i in range(len(test_base)):
            min_max(test_base[i],len(test_base[i])-1)

        knn(training_base,test_base,K)
    else:
        print("Usage: knn trainBase testBase K")


if __name__ == "__main__":
    main()

