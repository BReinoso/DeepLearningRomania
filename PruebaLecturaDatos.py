import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from random import shuffle
import time
import matplotlib.pyplot as plt
from theano import pp

CLEVELAND='processed.cleveland.data.txt'
HUNGARIAN='processed.hungarian.data.txt'
SWIT='processed.switzerland.data.txt'
VADATA='processed.va.data.txt'
DATASETS=['processed.cleveland.data.txt','processed.hungarian.data.txt',
          'processed.switzerland.data.txt','processed.va.data.txt']
def processLine(line):
    x = line.split(',')
    for i in range(len(x)):
        if x[i]=='?':
            x[i]=0
        else:
            x[i]= float(x[i])
    return x[:(len(x)-1)],x[(len(x)-1)]

def processFile(str):
    f=open(str,'r')
    x=[]
    y=[]
    for line in f:
        [temp1,temp2]=processLine(line)
        x.append(temp1)
        y.append(temp2)
    return x,y
def normalization(x):
    norm=[]
    maximum=[]
    minimum=[]
    for i in range(len(x[0])):
        maximum.append(max(x[:,i]))
        minimum.append(min(x[:,i]))
    for i in x:
        temp=[]
        for j in range(len(i)):
            temp.append((i[j]-minimum[j])/(maximum[j]-minimum[j]))
        norm.append(temp)
    return norm

if __name__ == "__main__":
    inputs=[]
    ouputs=[]
    for i in DATASETS:
        if i == CLEVELAND:
            [temp1,temp2]=processFile(CLEVELAND)
        elif i == HUNGARIAN:
            [temp1,temp2]=processFile(HUNGARIAN)
        elif i == SWIT:
            [temp1,temp2]=processFile(SWIT)
        elif i == VADATA:
            [temp1,temp2]=processFile(VADATA)
        inputs.extend(temp1)
        ouputs.extend(temp2)
    inputs=np.array(inputs)
    inputsNorm=normalization(inputs)
    w=np.array(np.random.rand(5,2), dtype=theano.config.floatX)
    x=np.array([0,1,2,3])
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    print(new_x.eval())
    print(w)
    m = T.dot(w.T, new_x)
    print(m.eval())
    a=np.array([0,1])
    b=np.array([0.3,0.8])
    c=np.array([0.8,0.3])
    print(a)
    print(b)
    print(c)
    print(1-a)
    print(-a*np.log(b))
    print(-(1-a)*np.log(1-b))
    print(-a*np.log(b)-(1-a)*np.log(1-b))
    print(np.sum(-a*np.log(b)-(1-a)*np.log(1-b)))
    a=[0,0,0,0]
    a+=[1,2]
    print(a)
    
    
