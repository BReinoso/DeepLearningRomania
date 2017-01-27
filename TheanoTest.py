import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from random import shuffle
import time
import matplotlib.pyplot as plt

CLEVELAND='processed.cleveland.data.txt'
HUNGARIAN='processed.hungarian.data.txt'
SWIT='processed.switzerland.data.txt'
VADATA='processed.va.data.txt'
DATASETS=['processed.cleveland.data.txt','processed.hungarian.data.txt',
          'processed.switzerland.data.txt','processed.va.data.txt']
LOG='./LOG/'

def normalization(x):
    norm=[]
    maximum=[]
    minimum=[]
    for i in range(len(x[0])):
        maximum.append(max(x[:][i]))
        minimum.append(min(x[:][i]))
    for i in x:
        temp=[]
        for j in range(len(i)):
            temp.append((i[j]-minimum[j])/(maximum[j]-minimum[j]))
        norm.append(temp)
    return norm

def processLine(line):
    x = line.split(',')
    for i in range(len(x)):
        if x[i]=='?':
            x[i]=0
        else:
            x[i]= float(x[i])
    return x

def processFile(str):
    f=open(str,'r')
    x=[]
    for line in f:
        x.append(processLine(line))
    f.close()
    for i in x:
        if i[len(i)-1] > 1:
            i=1
    return x

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    h = nnet.sigmoid(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.01 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

def getData():
    data=[]
    inputs=[]
    ouputs=[]
    for i in DATASETS:
        if i == CLEVELAND:
            data.extend(processFile(CLEVELAND))
        elif i == HUNGARIAN:
            data.extend(processFile(HUNGARIAN))
        elif i == SWIT:
            data.extend(processFile(SWIT))
        elif i == VADATA:
            data.extend(processFile(VADATA))
    shuffle(data)
    for i in data:
        inputs.append(i[:len(i)-1])
        ouputs.append(i[len(i)-1])
    inputs=normalization(inputs)
    return inputs,ouputs
    

if __name__ == "__main__":
    f=open(LOG+str(time.strftime("%d_%m_%Y")),'a')
    f.write('LOG for training started at: '+ str(time.strftime("%H:%M:%S"))+'\n')
    [inputs,ouputs]=getData()
    x = T.dvector()
    y = T.dscalar()
    fci=T.dscalar()
    thi=T.matrix()
    theta1 = theano.shared(np.array(np.random.rand(14,7), dtype=theano.config.floatX))
    theta2 = theano.shared(np.array(np.random.rand(8,4), dtype=theano.config.floatX))
    theta3 = theano.shared(np.array(np.random.rand(5,1), dtype=theano.config.floatX))
    hid1 = layer(x, theta1)
    hid2 = layer(hid1, theta2)
    out1 = T.sum(layer(hid2, theta3)) #output layer
    fc = ((out1 - y)**2)/2 #cost expression
    cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),
        (theta2, grad_desc(fc, theta2)),
        (theta3, grad_desc(fc, theta3))])
    cost1 = theano.function(inputs=[x, y], outputs=fc)
    run_forward = theano.function(inputs=[x], outputs=out1)
    lenT=int(len(inputs)*0.7)
    inputsTr=inputs[:lenT]
    inputsTe=inputs[lenT:]
    ouputsTr=ouputs[:lenT]
    ouputsTe=ouputs[lenT:]
    inputsTraining = np.array(inputsTr).reshape(len(inputsTr),13) #training data X
    ouputsTraining = np.array(ouputsTr) #training data Y
    inputsTest = np.array(inputsTe).reshape(len(inputsTe),13) #test data X
    ouputsTest = np.array(ouputsTe) #test data Y
    plt.axis([0, 10000, 0, 1])
    plt.ion()
    for i in range(10000):
        costTraining=0
        costTest=0
        if i % 50 == 0:
            for j in range(len(inputsTraining)):
                costTraining+=cost1(inputsTraining[j], ouputsTraining[j])
            for z in range(len(inputsTest)):
                costTest+=cost1(inputsTest[z], ouputsTest[z])
            costTraining/=len(inputsTraining)
            costTest/=len(inputsTest)
            plt.scatter(i, costTraining,color='r')
            plt.scatter(i, costTest,color='b')
            plt.pause(0.05)
            f.write('Epoch;'+str(i)+';costTraining;'+str(costTraining)+';costTest;'+str(costTest)+'\n')
        for k in range(len(inputsTraining)):
            cost(inputsTraining[k], ouputsTraining[k])
    f.close
