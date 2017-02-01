#Atuhors: Bryan Reinoso Cevallos y Alvaro Sainz Barcena
#Date: 28/02/2017
#Synopsis: Neural Network wirh 3 layers of 10,5 and 2 neurons. Using heart disease dataset
#Dataset: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
#------------------------------------------------
#Requirement: Folders .\LOG and .\PLOT must exist
#------------------------------------------------

#IMPORT LIBRARIES
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from random import shuffle
import time
import matplotlib.pyplot as plt

#DATASET NAMES AND LOG FOLDER
DATASETS=['processed.cleveland.data.txt','processed.hungarian.data.txt',
          'processed.switzerland.data.txt','processed.va.data.txt']
LOG='./LOG/'
PLOT='./PLOT/'

#Description: Method that applies mean normalization on the features of all examples in
#             in the input.
#Input: Array with input examples in each row
#Ouput: Array with the same examples but with its features normalized
def normalization(x):
    norm=[]
    maximum=[]
    minimum=[]
    #Taking the max and min for each feature
    for i in range(len(x[0])):
        maximum.append(max(x[:,i])) #Max of a colum of numpy array
        minimum.append(min(x[:,i])) #Min of a colum of numpy array
    #Applying the mean normalization
    for i in x:
        temp=[]
        for j in range(len(i)):
            temp.append((i[j]-minimum[j])/(maximum[j]-minimum[j])) #Mean normalization
        norm.append(temp) #Building a new normalized array
    return norm

#Description: Method that process a line read from the dataset file.
#
#Input: String with the line
#Ouput: Array with the line content ready to be treated
def processLine(line):
    x = line.split(',') #Splitting the values in different variables
    for i in range(len(x)):
        if x[i]=='?': #If the data is unknown we consider it as cero
            x[i]=0
        else:
            x[i]= float(x[i])
    return x

#Description: Method that process a file with a particular name.
#
#Input: String with the filename
#Ouput: Array with the examples of the dataset
def processFile(str):
    f=open(str,'r') #Open the file to be read
    x=[]
    for line in f:
        x.append(processLine(line)) #Prcessing lines and saving it in x
    f.close() #Close file
    for i in range(len(x)): #For values greater than 1 in the predicted value, we asume them as 1
        if x[i][len(x[i])-1] > 1:
            x[i][len(x[i])-1]=1
    return x

#Description: Method that calculate the result of pass the x input through the layer with w weights.
#
#Input: X->Input for the layer, W-> Layer's weights
#Ouput: The result of aplying the input through the neurons and the activaiton function
def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    h = nnet.sigmoid(m)
    return h

#Description: Method to get the data of all DATASETS.
#
#Input: String with the line
#Ouput: Array with the line content ready to be treated
def getData():
    data=[]
    inputs=[]
    outputs=[]
    outputsF=[]
    for i in DATASETS:
        data.extend(processFile(i)) #Storing the data of all datasets
    shuffle(data) #Shuffle the data of all datasets
    for i in data:
        inputs.append(i[:len(i)-1]) #Split the inputs
        outputs.append(i[len(i)-1]) #Split the expected outputs
    for i in outputs:                   #Converting the outputs to be predicted by 2 neurons
        if i==1:
            outputsF.append([1,0])
        else:
            outputsF.append([0,1])
    return inputs,outputsF

#Description: Method tha calculate the new theta.
#
#Input: cost-> CostFunciont from where to extrac the Gradient
#       theta->The weights to be updated
#       alpha->Learning rate
#Ouput: New theta updated
def gradient(cost,theta,alpha):
    return theta - alpha*T.grad(cost, wrt=theta)

#Description: Main thread.
#   Here the training and other process are done.
if __name__ == "__main__":
    name=str(time.strftime("%d_%m_%Y")) #Log file name
    time=str(time.strftime("%H_%M"))    #Strat time for execution
    f=open(LOG+name,'a')                #Open and create if its neccessary the log file
    alpha=0.1                           #Learning Rate
    num_Epochs=20                       #Number of Epochs for the training
    plotRate=1                          #Every x we store the data to be plotted and written in the log files
    f.write('Time;'+ time +';Epochs;'+str(num_Epochs)+';Alpha;'+str(alpha)+'\n') #First Line of the log of every train
    [inputs,outputs]=getData()          #Getting the data
    x = T.dvector('x')                  #Tensor variable to represent the example input
    y = T.dvector('y')                  #Tensor variable to represent the expected value
    #Shared variable of Weights of layer1. 14 inputs in every neuron (13 features examples + 1 bias) and 10 neurons.
    theta1 = theano.shared(np.array(np.random.rand(14,10), dtype=theano.config.floatX))
    #Shared variable of Weights of layer2. 11 inputs in every neuron (10 features examples + 1 bias) and 5 neurons.
    theta2 = theano.shared(np.array(np.random.rand(11,5), dtype=theano.config.floatX))
    #Shared variable of Weights of layer3. 6 inputs in every neuron (5 features examples + 1 bias) and 2 neurons.
    theta3 = theano.shared(np.array(np.random.rand(6,2), dtype=theano.config.floatX))
    #Getting the 70% of data to training and 30% to test
    lenT=int(len(inputs)*0.7)
    inputs=np.array(inputs)         #To normalize is neccessary the data to be stored in numpy array
    inputs=normalization(inputs)    #Normalization of the inputs
    inputsTr=inputs[:lenT]
    inputsTe=inputs[lenT:]
    outputsTr=outputs[:lenT]
    outputsTe=outputs[lenT:]
    #Converting the previous data into numpy arrays
    inputsTraining = np.array(inputsTr).reshape(len(inputsTr),13) #training data X
    outputsTraining = np.array(outputsTr).reshape(len(inputsTr),2) #training data Y
    inputsTest = np.array(inputsTe).reshape(len(inputsTe),13) #test data X
    outputsTest = np.array(outputsTe).reshape(len(outputsTe),2) #test data Y
    #Defining the outputs  of every layer as a Theano expression
    layer1=layer(x,theta1)
    layer2=layer(layer1,theta2)
    layer3=layer(layer2,theta3)
    #Implementing the function to get the predicted value
    predict=theano.function(inputs=[x],outputs=layer3)
    #Theano expression of the cost function formula
    cost_value=T.sum(-y*T.log(layer3)-(1-y)*T.log(1-layer3))
    #Implementing the cost function to train the model. Theta update
    cost_function=theano.function(inputs=[x,y],outputs=[cost_value],updates=[(theta1,gradient(cost_value,theta1,alpha)),(theta2,gradient(cost_value,theta2,alpha)),(theta3,gradient(cost_value,theta3,alpha))])
    #Implementing the cost function to test. No theta update
    cost_function_test=theano.function(inputs=[x,y],outputs=[cost_value])
    #Defining the plots
    fig=plt.figure()
    ax1=fig.add_subplot(311)
    ax1.set_title("Cost of all examples")
    ax1.set_ylabel("Cost value")
    ax1.set_xlabel("Epoch")
    ax3=fig.add_subplot(312)
    ax3.set_title("Hits/Misses Training Set per Epoch")
    ax3.set_ylabel("% Hits/Misses")
    ax3.set_xlabel("Epoch")
    ax4=fig.add_subplot(313)
    ax4.set_title("Hits/Misses Test Set per Epoch")
    ax4.set_ylabel("% Hits/Misses")
    ax4.set_xlabel("Epoch")
    plt.tight_layout()# Moving the graphs
    #Creating variables to store the data of the hits and misses in the epochs ->num_Epochs%plotRate==0
    numHitsCont=int(num_Epochs/plotRate)
    hitTotalTr=np.zeros(numHitsCont)
    missTotalTr=np.zeros(numHitsCont)
    hitTotalTe=np.zeros(numHitsCont)
    missTotalTe=np.zeros(numHitsCont)
    #Starting the training
    for i in range(num_Epochs):
        totalCost=0                             #Cost of training in the current epoch
        for j in range(len(inputsTr)):          #Training in every example with one update for each example
            [costIt]=cost_function(inputsTr[j],outputsTr[j])
            totalCost+=costIt
        if num_Epochs%plotRate==0:              #Deciding to plot and store data
            testCost=0
            for z in range(len(inputsTe)):      #Calculating the cost function of the test set
                [temp]=cost_function_test(inputsTe[z],outputsTe[z])
                testCost+=temp
            testCost/=len(inputsTe)             #Test cost function
            ax1.scatter(i, testCost,color='b')  #Plot test cost
            totalCost/=len(inputsTr)            #Training cost function
            f.write('Epoch;'+str(i)+';costTraining;'+str(totalCost)+';costTest;'+str(testCost)+'\n') #Writting the data on log file
            ax1.scatter(i, totalCost,color='r') #Plot training cost
            plt.pause(0.05)                     #Plot pause to print
            indexL=int(i/plotRate)              #Index for the hits and misses to be stored
            for hitTr in range(len(inputsTr)):  #Storing the hits and misses of training set in the current epoch
                prediction=predict(inputsTr[hitTr])
                predictionR=np.around(prediction)
                if (predictionR==outputsTr[hitTr]).all():
                    hitTotalTr[indexL]=hitTotalTr[indexL]+1
                else:
                    missTotalTr[indexL]=missTotalTr[indexL]+1
            percentHitTr=hitTotalTr[indexL]/len(inputsTr)
            #Plot for the second graph
            ax3.scatter(i,percentHitTr,c='g')
            percentMissTr=missTotalTr[indexL]/len(inputsTr)
            ax3.scatter(i,percentMissTr,c='r')
            for hitTe in range(len(inputsTe)):  #Storing the hits and misses of test set in the current epoch
                prediction=predict(inputsTe[hitTe])
                predictionR=np.around(prediction)
                if (predictionR==outputsTe[hitTe]).all():
                    hitTotalTe[indexL]=hitTotalTe[indexL]+1
                else:
                    missTotalTe[indexL]=missTotalTe[indexL]+1
            #Plot for the third graph
            percentHitTe=hitTotalTe[indexL]/len(inputsTe)
            ax4.scatter(i,percentHitTe,c='g')
            percentMissTe=missTotalTe[indexL]/len(inputsTe)
            ax4.scatter(i,percentMissTe,c='r')
    f.write('********************END********************\n') #Printing end in log file
    f.close                                     #close log file
    f=open(LOG+name+'_HitsMisses','a')          #Open or create of log file for hits and misses
    f.write('Time;'+ time +';Epochs;'+str(num_Epochs)+';Alpha;'+str(alpha)+'\n')
    for i in range(len(hitTotalTr)):            #Printting the hits and misses stored during the training
        s1='Epoch;'+str(i*plotRate)
        s2=';Hits Training;'+str(hitTotalTr[i])+';Misses Training;'+str(missTotalTr[i])
        s3=';Hits Test;'+str(hitTotalTe[i])+';Misses Test;'+str(missTotalTe[i])+'\n'
        f.write(s1+s2+s3)
    #Calculating the test hits and misses after the training
    hits=0
    misses=0
    for i in range(len(inputsTe)):
        prediction=predict(inputsTe[i])
        predictionR=np.around(prediction)
        if (predictionR==outputsTe[i]).all():
            hits+=1
        else:
            misses+=1
    #Writting the end line in log file
    f.write('-------------------TRAINING ENDED-------------------\n')
    #Writting the test hits and misses after the training in log file
    f.write('Test Hits: '+str(hits)+' Test misses: '+str(misses)+'\n')
    #Printing
    print('Test Hits: '+str(hits)+' Test misses: '+str(misses))
    #Calculating the training hits and misses after the training
    hits=0
    misses=0
    for i in range(len(inputsTr)):
        prediction=predict(inputsTr[i])
        predictionR=np.around(prediction)
        if (predictionR==outputsTr[i]).all():
            hits+=1
        else:
            misses+=1
    #Writting the training hits and misses after the training in log file
    f.write('Training Hits: '+str(hits)+' Training misses: '+str(misses)+'\n')
    #Printing
    print('Training Hits: '+str(hits)+' Training misses: '+str(misses))
    f.write('********************END********************\n') #End of log
    f.close #Close of log
    #Mantaining the plot
    plt.savefig(PLOT+'Plot_'+name+'_'+time+'.png') #Saving the graphs as a png image
    while True:
        try:
            plt.pause(0.05)
        except KeyboardInterrupt: #Capturing the exception to close the program
            break






