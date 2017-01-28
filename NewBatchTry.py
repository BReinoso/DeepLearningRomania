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

def getData():
    data=[]
    inputs=[]
    outputs=[]
    outputsF=[]
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
        outputs.append(i[len(i)-1])
    for i in outputs:
        if i==1:
            outputsF.append([1,0])
        else:
            outputsF.append([0,1])
    return inputs,outputsF

def gradient(cost,theta,alpha):
    return theta - alpha*T.grad(cost, wrt=theta)


if __name__ == "__main__":
    name=str(time.strftime("%d_%m_%Y"))
    time=str(time.strftime("%H:%M"))
    f=open(LOG+name,'a')
    alpha=0.1
    num_Epochs=20
    plotRate=1
    f.write('Time;'+ time +';Epochs;'+str(num_Epochs)+';Alpha;'+str(alpha)+'\n')
    [inputs,outputs]=getData()
    x = T.dvector('x')
    y = T.dvector('y')
    theta1 = theano.shared(np.array(np.random.rand(14,10), dtype=theano.config.floatX))
    theta2 = theano.shared(np.array(np.random.rand(11,5), dtype=theano.config.floatX))
    theta3 = theano.shared(np.array(np.random.rand(6,2), dtype=theano.config.floatX))
    lenT=int(len(inputs)*0.7)
    inputs=np.array(inputs)
    inputs=normalization(inputs)
    inputsTr=inputs[:lenT]
    inputsTe=inputs[lenT:]
    outputsTr=outputs[:lenT]
    outputsTe=outputs[lenT:]
    inputsTraining = np.array(inputsTr).reshape(len(inputsTr),13) #training data X
    outputsTraining = np.array(outputsTr).reshape(len(inputsTr),2) #training data Y
    inputsTest = np.array(inputsTe).reshape(len(inputsTe),13) #test data X
    outputsTest = np.array(outputsTe).reshape(len(outputsTe),2) #test data Y
    layer1=layer(x,theta1)
    layer2=layer(layer1,theta2)
    layer3=layer(layer2,theta3)
    predict=theano.function(inputs=[x],outputs=layer3)
    cost_value=T.sum(-y*T.log(layer3)-(1-y)*T.log(1-layer3))
    #cost_value
    cost_function=theano.function(inputs=[x,y],outputs=[cost_value],updates=[(theta1,gradient(cost_value,theta1,alpha)),(theta2,gradient(cost_value,theta2,alpha)),(theta3,gradient(cost_value,theta3,alpha))])
    cost_function_test=theano.function(inputs=[x,y],outputs=[cost_value])
    plt.axis([0, num_Epochs, 0, 2])
    plt.ion()
    numHitsCont=int(num_Epochs/plotRate)
    hitTotalTr=np.zeros(numHitsCont)
    missTotalTr=np.zeros(numHitsCont)
    hitTotalTe=np.zeros(numHitsCont)
    missTotalTe=np.zeros(numHitsCont)
    for i in range(num_Epochs):
        totalCost=0
        for j in range(len(inputsTr)):
            [costIt]=cost_function(inputsTr[j],outputsTr[j])
            totalCost+=costIt
        if num_Epochs%plotRate==0:
            testCost=0
            for z in range(len(inputsTe)):
                [temp]=cost_function_test(inputsTe[z],outputsTe[z])
                testCost+=temp
            testCost/=len(inputsTe)
            plt.scatter(i, testCost,color='b')
            totalCost/=len(inputsTr)
            f.write('Epoch;'+str(i)+';costTraining;'+str(totalCost)+';costTest;'+str(testCost)+'\n')
            plt.scatter(i, totalCost,color='r')
            plt.pause(0.05)
            indexL=int(i/plotRate)
            for hitTr in range(len(inputsTr)):
                prediction=predict(inputsTr[hitTr])
                predictionR=np.around(prediction)
                if (predictionR==outputsTr[hitTr]).all():
                    hitTotalTr[indexL]=hitTotalTr[indexL]+1
                else:
                    missTotalTr[indexL]=missTotalTr[indexL]+1
            for hitTe in range(len(inputsTe)):
                prediction=predict(inputsTe[hitTe])
                predictionR=np.around(prediction)
                if (predictionR==outputsTe[hitTe]).all():
                    hitTotalTe[indexL]=hitTotalTe[indexL]+1
                else:
                    missTotalTe[indexL]=missTotalTe[indexL]+1
    f.write('********************END********************\n')
    f.close
    f=open(LOG+name+'_HitsMisses','a')
    f.write('Time;'+ time +';Epochs;'+str(num_Epochs)+';Alpha;'+str(alpha)+'\n')
    for i in range(len(hitTotalTr)):
        s1='Epoch;'+str(i*plotRate)
        s2=';Hits Training;'+str(hitTotalTr[i])+';Misses Training;'+str(missTotalTr[i])
        s3=';Hits Test;'+str(hitTotalTe[i])+';Misses Test;'+str(missTotalTe[i])+'\n'
        f.write(s1+s2+s3)
    hits=0
    misses=0
    for i in range(len(inputsTe)):
        prediction=predict(inputsTe[i])
        predictionR=np.around(prediction)
        if (predictionR==outputsTe[i]).all():
            hits+=1
        else:
            misses+=1
    f.write('-------------------TRAINING ENDED-------------------\n')
    f.write('Test Hits: '+str(hits)+' Test misses: '+str(misses)+'\n')
    print('Test Hits: '+str(hits)+' Test misses: '+str(misses))
    hits=0
    misses=0
    for i in range(len(inputsTr)):
        prediction=predict(inputsTr[i])
        predictionR=np.around(prediction)
        if (predictionR==outputsTr[i]).all():
            hits+=1
        else:
            misses+=1
    f.write('Training Hits: '+str(hits)+' Training misses: '+str(misses)+'\n')
    print('Training Hits: '+str(hits)+' Training misses: '+str(misses))
    f.write('********************END********************\n')
    f.close
    while True:
        plt.pause(0.05)





