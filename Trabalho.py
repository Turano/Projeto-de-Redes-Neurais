import math
import numpy as np

#Função de Custo
def CrossEntropy(out, value):
    if value ==1:
        return -np.log(out)
    else:
        return -np.log(1 - out)

# Esquece essa porra
def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num


# Declaração do modelo do perceptron
def perc(num, peso, bias):
    return 1/(1+(math.exp(-(sum(multiplicar(num, peso), bias)))))
    

# Abro o arquivo de imagem
f = open("semeion.data", "r")

list = f.readline().split(" ")

input = np.zeros(256)

#Converto os dados de lista para float
for x in range (256):
    input[x] = float(list[x])

#Declaro os pesos iniciais, por meio da função de He-et-al
whl = np.random.randn(16, 256) * np.sqrt(2 / 256)

wout = np.random.randn(10, 16) * np.sqrt(2 / 16)

#Declaro a hidden layer
hl = np.zeros(16)

#Declaro o output
output = np.zeros(10)

cost = np.zeros(10)

bias = 0

#Passo os valores pelos perceptrons
for x in range (16):
    hl[x] = perc(input, whl[x], bias)

for x in range (10):
    output[x] = perc(hl, wout[x], bias)

#print("Output antes do Cross Entropy:\n", output)

#Aplico a função de custo
for x in range (10):
    cost[x] = CrossEntropy(output[x], list[x+256])

#print("Custo:\n", cost)

gradient = np.gradient(wout)

print(gradient[0][0])

#Fecho o arquivo
f.close()