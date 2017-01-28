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
        maximum.append(max(x[:,i]))
        minimum.append(min(x[:,i]))
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
    for i in range(len(x)):
        if x[i][len(x[i])-1] > 1:
            x[i][len(x[i])-1]=1
    return x

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    h = nnet.sigmoid(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.03 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

def getData():
    data=[]
    inputs=[]
    ouputs=[]
    ouputsF=[]
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
    for i in ouputs:
        if i==1:
            ouputsF.append([1,0])
        else:
            ouputsF.append([0,1])
    return inputs,ouputsF

def gradient(cost,theta):
    return T.grad(cost, wrt=theta)

if __name__ == "__main__":
    lamba=1
    alpha=0.1
    f=open(LOG+str(time.strftime("%d_%m_%Y")),'a')
    f.write('LOG for training started at: '+ str(time.strftime("%H:%M:%S"))+'\n')
    [inputs,ouputs]=getData()
    x = T.dvector()
    y = T.dvector()
    theta1 = T.matrix()
    #theta1 = theano.shared(np.array(np.random.rand(14,15), dtype=theano.config.floatX))
    theta2 = T.matrix()
    #theta2 = theano.shared(np.array(np.random.rand(16,10), dtype=theano.config.floatX))
    theta3 = T.matrix()
    #theta3 = theano.shared(np.array(np.random.rand(11,5), dtype=theano.config.floatX))
    theta4 = T.matrix()
    #theta4 = theano.shared(np.array(np.random.rand(6,1), dtype=theano.config.floatX))
    hid1 = layer(x, theta1)
    hid2 = layer(hid1, theta2)
    hid3 = layer(hid2, theta3)
    out1 = layer(hid3, theta4) #output layer
    fc = ((out1 - y)**2)/2 #cost expression
    cost_temp=np.array(-y*np.log(out1)-(1-y)*np.log(1-out1))
    cost_function=np.sum(cost_temp)
    #cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    #    (theta1, grad_desc(fc, theta1)),
    #    (theta2, grad_desc(fc, theta2)),
    #    (theta3, grad_desc(fc, theta3)),
    #    (theta4, grad_desc(fc, theta4))])
    cost1 = theano.function(inputs=[x, y], outputs=cost_function)
    run_forward = theano.function(inputs=[x], outputs=out1)
    lenT=int(len(inputs)*0.7)
    inputs=np.array(inputs)
    inputs=normalization(inputs)
    inputsTr=inputs[:lenT]
    inputsTe=inputs[lenT:]
    ouputsTr=ouputs[:lenT]
    ouputsTe=ouputs[lenT:]
    theta1_1=np.array(np.random.rand(14,15), dtype=theano.config.floatX)
    theta2_2=np.array(np.random.rand(16,10), dtype=theano.config.floatX)
    theta3_3=np.array(np.random.rand(11,5), dtype=theano.config.floatX)
    theta4_4=np.array(np.random.rand(6,2), dtype=theano.config.floatX)
    inputsTraining = np.array(inputsTr).reshape(len(inputsTr),13) #training data X
    ouputsTraining = np.array(ouputsTr).reshape(len(ouputsTr),2)#training data Y
    inputsTest = np.array(inputsTe).reshape(len(inputsTe),13) #test data X
    ouputsTest = np.array(ouputsTe).reshape(len(ouputsTr),2)#test data Y
    plt.axis([0, 10000, 0, 1])
    plt.ion()
    for i in range(10000):
        grad_act=[0,0,0,0]
        costTraining=0
        costTest=0
        if i % 50 == 0:
            for j in range(len(inputsTraining)):
                costTraining+=cost1(inputsTraining[j], ouputsTraining[j])
            for z in range(len(inputsTest)):
                costTest+=cost1(inputsTest[z], ouputsTest[z])
            costTraining/=len(inputsTraining)
            #costTraining+=((lamba/len(inputsTraining))*
            costTest/=len(inputsTest)
            plt.scatter(i, costTraining,color='r')
            plt.scatter(i, costTest,color='b')
            plt.pause(0.05)
            f.write('Epoch;'+str(i)+';costTraining;'+str(costTraining)+';costTest;'+str(costTest)+'\n')
        for k in range():
            cost=cost1(inputsTraining[k], )
            grad_act[0]+=gradient(cost,theta1_1)
            grad_act[1]+=gradient(cost,theta2_2)
            grad_act[2]+=gradient(cost,theta3_3)
            grad_act[3]+=gradient(cost,theta4_4)
        for i in range(len(grad_act)):
            grad_act[i]/=len(inputsTraining)
        theta1_1-=alpha*grad_act[0]
        theta2_2-=alpha*grad_act[1]
        theta3_3-=alpha*grad_act[2]
        theta4_4-=alpha*grad_act[3]
                
    f.close
