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
        inputs.append(temp1)
        ouputs.append(temp2)
    lenT=int(len(inputs)*0.7)
    inputsTr=inputs[:lenT]
    inputsTe=inputs[lenT:]
    ouputsTr=ouputs[:lenT]
    ouputsTe=ouputs[lenT:]
    
