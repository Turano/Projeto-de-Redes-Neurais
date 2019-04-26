import math
import numpy as np

# Modelo do perceptron
def perc(num, peso, wbias):
    return 1/(1+(math.exp(-(sum(multiplicar(num, peso), wbias)))))

def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num

def feedforward(whl, wout, hl, outnosig, hlnosig, output, input, wbhl, wbout):

    transpostahl = np.matrix.transpose(whl)

    transpostaout = np.matrix.transpose(wout)

    for x in range (30):
        hl[x] = perc(input, transpostahl[x], wbhl[x])
        hlnosig[x] = sum(multiplicar(input, transpostahl[x]), wbhl[x])

    for x in range (10):
        output[x] = perc(hl, transpostaout[x], wbout[x])
        outnosig[x] = sum(multiplicar(hl, transpostaout[x]), wbout[x])

    return (output, outnosig, hlnosig)

def backpropagation(output, algarismo, wout, whl, input, hl, outnosig, hlnosig, wbhl, wbout):

    taxa = 0.3

    cost = 0
    costfunction = 0

    awout = wout
    awhl = whl

    for x in range (10):
        costfunction += ((output[x] - algarismo[x])**2)/2
        cost += output[x] - algarismo[x]

    for x in range(30):
        for y in range(10): 
            awout[x][y] = wout[x][y] - taxa * cost * (1/(1+np.exp(-(hl[x] * wout[x][y] + wbout[y])))) * (1 - 1/(1+np.exp(-(hl[x] * wout[x][y] + wbout[y])))) * hl[x]

    for x in range(256):
        for y in range(30):
            awhl[x][y] = whl[x][y] - taxa * cost * (1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * (1 - 1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * sum(wout[y]) * (1/(1+np.exp(-(input[x]* whl[x][y] + wbhl[y])))) * (1 - 1/(1+np.exp(-(input[x]* whl[x][y] + wbhl[y])))) * input[x]

    for x in range (10):    
        wbout[x] = wbout[x] - taxa * cost * (1/(1+np.exp(-sum(outnosig)))) * (1 - 1/(1+np.exp(-sum(outnosig))))
    for x in range (30):
        wbhl[x] = wbhl[x] - taxa * cost * (1/(1+np.exp(-sum(outnosig)))) * (1 - 1/(1+np.exp(-sum(outnosig)))) * sum(wout[x]) * (1/(1+np.exp(-sum(hlnosig)))) * (1 - 1/(1+np.exp(-sum(hlnosig))))

    return (costfunction, awhl, awout, wbhl, wbout)

def main():
    error = open("error", "a+")

    input = np.zeros(256)
    algarismo = np.zeros(10)

    whl = np.random.randn(256, 30) * np.sqrt(2 / 256)
    wout = np.random.randn(30, 10) * np.sqrt(2 / 30)
    wbhl = np.random.random(30)
    wbout = np.random.random(10)

    hl = np.zeros(30)
    output = np.zeros(10)

    outnosig = np.zeros(10)
    hlnosig = np.zeros(30)

    erro = np.zeros(1593)

    for x in range(10):
        data = open("semeion.data", "r")

        for z in range(1593):
            adata = data.readline().split(" ")
            for x in range (256):
                input[x] = float(adata[x])
            for x in range (10):
                algarismo[x] = int(adata[256+x])
            (output, outnosig, hlnosig) = feedforward(whl, wout, hl, outnosig, hlnosig, output, input, wbhl, wbout)
            (erro[z], whl, wout, wbhl, wbout) = backpropagation(output, algarismo, wout, whl, input, hl, outnosig, hlnosig, wbhl, wbout)

        data.close()
        error.write(str(sum(erro)/1593))
        error.write("\n")
        print(str(sum(erro)/1593))
    error.close()

if __name__ == '__main__':
    main() 