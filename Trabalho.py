import numpy as np
import time
import os
import pickle
import random

def LReLu(out):
    if (out >= 0):
        return out
    else:
        return out*0.01

def perc(num, peso, wbias):
    return sum(multiplicar(num, peso), wbias)

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def multiplicar(num, peso):
    for x in range(len(num)):
        num[x] *= peso[x]
    return num

def feedforward(whl, wout, hl, output, entrada, wbhl, wbout):

    transpostahl = np.matrix.transpose(whl)

    transpostaout = np.matrix.transpose(wout)

    for x in range (60):
        hl[x] = LReLu(perc(entrada, transpostahl[x], wbhl[x]))

    for x in range (10):
        output[x] = (perc(hl, transpostaout[x], wbout[x]))

    output = softmax(output)

    return (output)

def backpropagation(output, algarismo, wout, whl, entrada, hl, wbhl, wbout, taxa):

    awout = wout
    awhl = whl
    aux1 = np.zeros(10)
    dercrosshl = np.zeros(60)
    dercrosssoft = np.zeros(10)
    awbout = np.zeros(10)
    awbhl = np.zeros(60)

    for x in range (10):
        dercrosssoft[x] = output[x] - algarismo[x]

    for y in range(60):
        for x in range(10):
            aux1[x] = dercrosssoft[x] * wout[y][x]
        dercrosshl[y] = sum(aux1)

    for x in range(60):
        for y in range(10):
            awout[x][y] = wout[x][y] - taxa * dercrosssoft[y] * hl[x]

    for x in range(256):
        for y in range(60):
            if(hl[y] >= 0):
                awhl[x][y] = whl[x][y] - taxa * dercrosshl[y] * entrada[x]
            else:
                awhl[x][y] = whl[x][y] - taxa * dercrosshl[y] * entrada[x] * 0.01

    for x in range (10):
        awbout[x] = wbout[x] - taxa * dercrosssoft[x]

    for x in range (60):
        if(hl[x] >= 0):
            awbhl[x] = wbhl[x] - taxa * dercrosshl[y]
        else:
            awbhl[x] = wbhl[x] - taxa * dercrosshl[y] * 0.01

    return (awhl, awout, awbhl, awbout)

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
            whl = np.random.randn(256, 60) * np.sqrt(2 / 256)
            wout = np.random.randn(60, 10) * np.sqrt(2 / 60)
            wbhl = np.random.randn(60) * np.sqrt(2 / 60)
            wbout = np.random.randn(10) * np.sqrt(2 / 10)
    else:
        print("ok")
        whl = np.random.randn(256, 60) * np.sqrt(2 / 256)
        wout = np.random.randn(60, 10) * np.sqrt(2 / 60)
        wbhl = np.random.randn(60) * np.sqrt(2 / 60)
        wbout = np.random.randn(10) * np.sqrt(2 / 10)

    hl = np.zeros(60)
    output = np.zeros(10)

    start_time = time.time()
    
    contagem = 0

    var = 1
    
    taxa01 = 0

    while(1):
        taxa = 0.0001

        for y in range (4):
            lines = open('Treinamento{}'.format(y+1)).readlines()
            random.shuffle(lines)
            open('Treinamento{}'.format(y+1), 'w').writelines(lines)
            with open('Treinamento{}'.format(y+1), 'r') as d:
                for z in range(286):
                    data = d.readline().split(" ")
                    for x in range (256):
                        entrada[x] = float(data[x])
                    for x in range (10):
                        algarismo[x] = int(data[256+x])
                    (output) = feedforward(whl, wout, hl, output, entrada, wbhl, wbout)
                    (whl, wout, wbhl, wbout) = backpropagation(output, algarismo, wout, whl, entrada, hl, wbhl, wbout, taxa)

            weights = [whl, wout, wbhl, wbout]

            file_name = os.path.join(os.path.abspath('pesos'), 'peso{:05d}'.format(contagem))
            with open(file_name, 'wb+') as p:
                pickle.dump(weights, p)

            contagem += 1
        cost = np.zeros(32)

        valor = -1

        with open('Treinamento5'.format(y), 'r') as d:
            for z in range (32):
                data = d.readline().split(" ")
                for x in range (256):
                    entrada[x] = float(data[x])
                for x in range (10):
                    algarismo[x] = int(data[256+x])
                (output) = feedforward(whl, wout, hl, output, entrada, wbhl, wbout)
                print(np.argmax(output))
                for x in range (10):    
                    cost[z] += algarismo[x] * (-np.log(output[x])) + (1 - algarismo[x]) * (-np.log(1 - output[x]))
                
                            
            
        with open('erro', 'a+') as e:
            e.write('{}\n'.format(sum(cost)/32))

        

if __name__ == '__main__':
    main()