import math
import numpy as np
import time
import os
import pickle

def perc(num, peso, wbias):
    return 1/(1+(math.exp(-(sum(multiplicar(num, peso), wbias)))))
def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num

def feedforward(whl, wout, hl, outnosig, hlnosig, output, entrada, wbhl, wbout):

    transpostahl = np.matrix.transpose(whl)

    transpostaout = np.matrix.transpose(wout)

    for x in range (30):
        hl[x] = perc(entrada, transpostahl[x], wbhl[x])
        hlnosig[x] = sum(multiplicar(entrada, transpostahl[x]), wbhl[x])

    for x in range (10):
        output[x] = perc(hl, transpostaout[x], wbout[x])
        outnosig[x] = sum(multiplicar(hl, transpostaout[x]), wbout[x])

    return (output, outnosig, hlnosig)

def backpropagation(output, algarismo, wout, whl, entrada, hl, outnosig, hlnosig, wbhl, wbout):

    taxa = 0.3

    cost = 0
    costfunction = 0

    awout = wout
    awhl = whl

    for x in range (10):
        #costfunction += ((output[x] - algarismo[x])**2)/2
        costfunction += -(algarismo[x] * np.log(output[x]))
        #cost += output[x] - algarismo[x]
        cost += -(algarismo[x] / output[x]) 

    for x in range(30):
        for y in range(10): 
            awout[x][y] = wout[x][y] - taxa * cost * (1/(1+np.exp(-(hl[x] * wout[x][y] + wbout[y])))) * (1 - 1/(1+np.exp(-(hl[x] * wout[x][y] + wbout[y])))) * hl[x]

    for x in range(256):
        for y in range(30):
            awhl[x][y] = whl[x][y] - taxa * cost * (1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * (1 - 1/(1+np.exp(-(hl[y]* sum(wout[y]))))) * sum(wout[y]) * (1/(1+np.exp(-(entrada[x]* whl[x][y] + wbhl[y])))) * (1 - 1/(1+np.exp(-(entrada[x]* whl[x][y] + wbhl[y])))) * entrada[x]

    for x in range (10):    
        wbout[x] = wbout[x] - taxa * cost * (1/(1+np.exp(-sum(outnosig)))) * (1 - 1/(1+np.exp(-sum(outnosig))))
    for x in range (30):
        wbhl[x] = wbhl[x] - taxa * cost * (1/(1+np.exp(-sum(outnosig)))) * (1 - 1/(1+np.exp(-sum(outnosig)))) * sum(wout[x]) * (1/(1+np.exp(-sum(hlnosig)))) * (1 - 1/(1+np.exp(-sum(hlnosig))))

    return (costfunction, awhl, awout, wbhl, wbout)

def main():

    entrada = np.zeros(256)
    algarismo = np.zeros(10)

    nome = input()
    if nome != "":
        try:
            print('ok')
            with open(os.path.join(os.path.abspath('pesos'), nome), 'rb') as n:
                whl, wout, wbhl, wbout = pickle.load(n)
        except:
            print("não ok")
            whl = np.random.randn(256, 30) * np.sqrt(2 / 256)
            wout = np.random.randn(30, 10) * np.sqrt(2 / 30)
            wbhl = np.random.random(30)
            wbout = np.random.random(10)
    else:
        print("ok")
    whl = np.random.randn(256, 30) * np.sqrt(2 / 256)
    wout = np.random.randn(30, 10) * np.sqrt(2 / 30)
    wbhl = np.random.random(30)
    wbout = np.random.random(10)

    hl = np.zeros(30)
    output = np.zeros(10)

    outnosig = np.zeros(10)
    hlnosig = np.zeros(30)

    erro = np.zeros(1593)

    start_time = time.time()
    
    contagem = 0

    while(True):
        with open('semeion.data', 'r') as d:
            for z in range(1593):
                data = d.readline().split(" ")
                for x in range (256):
                    entrada[x] = float(data[x])
                for x in range (10):
                    algarismo[x] = int(data[256+x])
                (output, outnosig, hlnosig) = feedforward(whl, wout, hl, outnosig, hlnosig, output, entrada, wbhl, wbout)
                (erro[z], whl, wout, wbhl, wbout) = backpropagation(output, algarismo, wout, whl, entrada, hl, outnosig, hlnosig, wbhl, wbout)

        with open('erro', 'a+') as e:
            e.write('{}\n'.format(sum(erro)/1593))

        weights = [whl, wout, wbhl, wbout]

        file_name = os.path.join(os.path.abspath('pesos'), 'peso{:03d}'.format(contagem))
        with open(file_name, 'wb+') as p:
            pickle.dump(weights, p)
            
        contagem += 1
           
        print('{:02d}:{:02d}:{:02d}\t{}'.format(int(time.time() - start_time) // 3600, (int(time.time() - start_time) % 3600 // 60), int(time.time() - start_time) % 60, sum(erro)/1593))

if __name__ == '__main__':
    main()