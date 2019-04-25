import math
import numpy as np

#Função de Custo MSE
def MSE(output, lista):
    return ((output - lista)**2)/2 

# Esquece
def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num


# Declaração do modelo do perceptron
def perc(num, peso, bias):
    return 1/(1+(math.exp(-(sum(multiplicar(num, peso), bias))))) 

#Declaro os pesos iniciais, por meio da função de He-et-al
whl = np.random.randn(256, 16) * np.sqrt(2 / 256)

wout = np.random.randn(16, 10) * np.sqrt(2 / 16)

awhl = np.random.randn(256, 16) * np.sqrt(2 / 256)

awout = np.random.randn(16, 10) * np.sqrt(2 / 16)

taxa = 0.2

erro = np.zeros(1593)

# Abro o arquivo de imagem
data = open("semeion.data", "r")
#weights = open("pesos", "w+")
error = open("error", "w+")


input = np.zeros(256)
algarismo = np.zeros(10)

#Declaro a hidden layer
hl = np.zeros(16)

#Declaro o output
output = np.zeros(10)

bias = 0

for z in range(1593):

    #np.savetxt(weights, wout)
    #np.savetxt(weights, whl)

    list = data.readline().split(" ")

    #Converto os dados de lista para float
    for x in range (256):
        input[x] = float(list[x])
    for x in range (10):
        algarismo[x] = int(list[256+x])

    cost = 0
    costfunction = 0

    transpostahl = np.matrix.transpose(whl)
    transpostaout = np.matrix.transpose(wout)

    #Passo os valores pelos perceptrons
    for x in range (16):
        hl[x] = perc(input, transpostahl[x], bias)

    for x in range (10):
        output[x] = perc(hl, transpostaout[x], bias)

    #Aplico a função de custo
    for x in range (10):
        costfunction += MSE(output[x], int(algarismo[x]))

    erro[z]= costfunction

    #error.write(str(costfunction))
    #error.write(" ")

    for x in range (10):
        cost += output[x] - int(algarismo[x])

    for x in range(16):
        for y in range(10): 
            awout[x][y] = wout[x][y] - taxa * (cost * (1/(1+np.exp(-(hl[x]* wout[x][y])))) * (1 - 1/(1+np.exp(-(hl[x]* wout[x][y])))) * hl[x])

    for x in range(256):
        for y in range(16):
            awhl[x][y] = whl[x][y] - taxa * (cost * (1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * (1 - 1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * sum(wout[y]) * (1 - 1/(1+np.exp(-(input[x]* whl[x][y])))) * input[x])

    wout = awout
    whl = awhl

error.write(str(sum(erro)/1593))

#Fecho o arquivo
data.close()

error.close()

#weights.close()