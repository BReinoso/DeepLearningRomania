import numpy as np
import matplotlib.pyplot as plt

y=range(100)
x=range(1,101)
x1=np.log(x)

fig=plt.figure()
ax1=fig.add_subplot(221)
ax1.set_title("Random")
ax2=fig.add_subplot(222)
ax2.set_title("Log")
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)

for i in range(10):
    temp = np.random.random()
    l=ax1.scatter(i,temp,s=10,c='b')
    plt.pause(0.05)
for i in range(len(x)):
    ax2.scatter(i,x[i],s=10,c='r')

ax1.legend([l],['Prueba'],loc='upper right')
while True:
    plt.pause(0.05)
