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
    minumum=[]
    for i in range(len(x[0])):
        maximum.append(max(x[:][i]))
        minimum.append(min(x[:][i]))
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
    inputsNorm=normalization(inputs)