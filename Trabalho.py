import numpy as np
import time
import os
import pickle
import random

np.seterr(all = 'raise')

def sigmoid(out):
    return 1/(1+np.exp(-out))

def perc(num, peso, wbias):
    return sum(multiplicar(num, peso), wbias)

def softmax(array):
    output = np.zeros(10)
    for x in range (10):
        output[x] = np.exp(array[x]) / sum(np.exp(array))
    return output

def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num

def feedforward(whl, wout, hl, hlnosig, output, entrada, wbhl, wbout):

    transpostahl = np.matrix.transpose(whl)

    transpostaout = np.matrix.transpose(wout)

    for x in range (30):
        hl[x] = sigmoid(perc(entrada, transpostahl[x], wbhl[x]))
        hlnosig[x] = sum(multiplicar(entrada, transpostahl[x]), wbhl[x])

    for x in range (10):
        output[x] = (perc(hl, transpostaout[x], wbout[x]))

    output = softmax(output)

    return (output, hlnosig)

def backpropagation(output, algarismo, wout, whl, entrada, hl, hlnosig, wbhl, wbout):

    taxa = 0.000001

    costfunction = 0

    awout = wout
    awhl = whl
    aux1 = np.zeros(10)
    dercrosshl = np.zeros(30)
    dercrosshlnosig = np.zeros(30)
    dercrosssoft = np.zeros(10)
    awbout = np.zeros(10)
    awbhl = np.zeros(30)

    for x in range (10):    
        costfunction += algarismo[x] * (-np.log(output[x])) + (1 - algarismo[x]) * (-np.log(1 - output[x]))

    #print(costfunction)

    for x in range (10):
        dercrosssoft[x] = output[x] - algarismo[x]

    for y in range(30):
        for x in range(10):
            aux1[x] = dercrosssoft[x] * wout[y][x]
        dercrosshl[y] = sum(aux1)
        dercrosshlnosig[y] = (1 / (1 + np.exp(-hlnosig[y]))) * (1 - (1 / (1 + np.exp(-hlnosig[y]))))

    for x in range(30):
        for y in range(10):
            awout[x][y] = wout[x][y] - taxa * dercrosssoft[y] * hl[x]

    for x in range(256):
        for y in range(30):
            awhl[x][y] = whl[x][y] - taxa * dercrosshl[y] * dercrosshlnosig[y] * entrada[x]

    for x in range (10):
        awbout[x] = wbout[x] - taxa * dercrosssoft[x]

    for x in range (30):
        awbhl[x] = wbhl[x] - taxa * dercrosshl[y] * dercrosshlnosig[y]

    return (costfunction, awhl, awout, awbhl, awbout)

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
            wbhl = np.random.random(30) * np.sqrt(2 / 30)
            wbout = np.random.random(10) * np.sqrt(2 / 10)
    else:
        print("ok")
        whl = np.random.randn(256, 30) * np.sqrt(2 / 256)
        wout = np.random.randn(30, 10) * np.sqrt(2 / 30)
        wbhl = np.random.random(30) * np.sqrt(2 / 30)
        wbout = np.random.random(10) * np.sqrt(2 / 10)

    hl = np.zeros(30)
    output = np.zeros(10)

    hlnosig = np.zeros(30)

    erro = np.zeros(1593)

    start_time = time.time()
    
    contagem = 0

    anterior = 0

    while(1):
        lines = open('semeion.data').readlines()
        random.shuffle(lines)
        open('semeion.data', 'w').writelines(lines)
        with open('semeion.data', 'r') as d:
            for z in range(1593):
                data = d.readline().split(" ")
                for x in range (256):
                    entrada[x] = float(data[x])
                for x in range (10):
                    algarismo[x] = int(data[256+x])
                (output, hlnosig) = feedforward(whl, wout, hl, hlnosig, output, entrada, wbhl, wbout)
                (erro[z], whl, wout, wbhl, wbout) = backpropagation(output, algarismo, wout, whl, entrada, hl, hlnosig, wbhl, wbout)

        with open('erro', 'a+') as e:
            e.write('{}\n'.format(sum(erro)/1593))

        weights = [whl, wout, wbhl, wbout]

        file_name = os.path.join(os.path.abspath('pesos'), 'peso{:03d}'.format(contagem))
        with open(file_name, 'wb+') as p:
            pickle.dump(weights, p)
            
        contagem += 1
           
        print('{:02d}:{:02d}:{:02d}\t{}\t{}'.format(int(time.time() - start_time) // 3600, (int(time.time() - start_time) % 3600 // 60), int(time.time() - start_time) % 60, sum(erro)/1593, anterior - sum(erro)/1593))

        anterior = sum(erro/1593)

if __name__ == '__main__':
    main()
