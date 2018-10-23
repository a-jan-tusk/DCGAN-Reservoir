#Using echo state networks for MNIST classification


from numpy import *
import pickle, gzip
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import itertools
from sklearn.decomposition import PCA
from scipy import signal
import random
import json
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
from matplotlib import ticker
import math
#import networkx as nx
#from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import math

with gzip.open('mnist.pkl.gz', 'rb') as f:
     train_set, valid_set, test_set = pickle.load(f,encoding='latin1')

f.close()#data= loadtxt('logistic.txt')

random.seed(42)

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def plot_mnist_digit(image):
    """ Plot a single MNIST image."""

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    xticks(np.array([]))
    yticks(np.array([]))
    show()

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

insize=28
start=0
dt = int(len(train_set[0][1])/insize)



siz=500


#generate a reservoir

lam=1.0
adj_res=(scipy.sparse.rand(siz, siz, density=0.90, format='coo', random_state=100).A-0.5)*2
adj_res[adj_res==-1]=0


adj_inp=(scipy.sparse.rand(siz, insize+1, density=0.90, format='coo', random_state=100).A-0.5)*2
adj_inp[adj_inp==-1]=0

#spectral radius (largest eigenvalue of adj matrix) for res system ~ 0.9

rho= math.sqrt(max(abs(linalg.eig(adj_res)[0])))
sr=0.9
adj_res *= sr/ rho

#training

tr_s=500  #training size
te_s=100  #test size

for i in range(tr_s):  
     a=random.randint(0,len(train_set[0]-1))
     data1 = train_set[0][a]
     data1=data1.reshape(dt,insize)
     val1=train_set[1][a]
     x1=zeros((siz,1))



     #run the reservoir
     err=1e-8

     for t in range(dt+start):
             a1=zeros((dt,1))
             a1[:,0]=data1[:,t]
             x1=lam*tanh(dot(adj_inp,vstack((1,a1))) + dot(adj_res,x1)) #reservoir update step
             if t==start:
                 Xr=abs(x1) #choice between abs(x1) and x1
             else:
                 Xr=hstack((Xr,abs(x1)))   #choice between abs(x1) and x1
     y=zeros((10,dt))
     y[val1,:]=1 


     if i==0:
             X=Xr
             Y=y
     else:
             X=hstack((X,Xr))
             Y=hstack((Y,y))


#regression 
clf = KernelRidge(alpha=1)
clf.fit(X.T, Y.T) 
Wpred = dot(dot(Y,X.T), linalg.inv(dot(X,X.T) + err*eye(dot(X,X.T).shape[0]) ) )
print ('finished_training')

#plot some reservoir activity and output weights

'''          for s in range(6):
               N=4
               nodes=50
               M=imshow(Hii[s][:nodes,:N*dt],  vmax=abs(X[30:][:]).max(),vmin=-abs(X[30:][:]).max(), interpolation='nearest', aspect='auto', origin='upper')
               cb=colorbar(M,shrink=0.5, pad=.1)
               cb.set_label('Reservoir  Activity', fontsize=10)
               ylabel('$Reservoir  Nodes$', fontsize=13)
               xlabel('$Time $', fontsize=13)
               #title('Reservoir activity', fontsize=20)
               plt.savefig('res_activity' +str(s)+'.jpg',pad_inches=1)
               plt.close()

               W=imshow(Wpred[:,:nodes], interpolation='nearest', aspect='auto', origin='upper')
               yint=[0,1,2,3,4,5]
               plt.yticks(yint)
               cb=colorbar(W,shrink=0.3, pad=.1)
               cb.set_label('Weights', fontsize=10)
               ylabel('$Readout  Nodes$', fontsize=13)
               xlabel('$ Reservoir  Nodes)$', fontsize=13)
               #title('', fontsize=20)
               plt.savefig('Wout'+str(nodes)+'nodes.jpg',pad_inches=1)
               plt.close()


               colors=cm.rainbow(np.linspace(0, 1, len(yint)))
               mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',colors)'''


 



#prediction state
correct=0
for i in range(te_s):
        posout=random.randint(0,len(test_set[0])-1)

        testdat1= test_set[0][posout].reshape(dt,insize)
        testval1=test_set[1][posout]

        xout1=zeros((siz,1))
        for t in range(dt+start):
                aout1=zeros((dt,1))
                aout1[:,0]=testdat1[:,t] 
                xout1=lam*tanh(dot(adj_inp,vstack((1,aout1))) + dot(adj_res,xout1)) #reservoir update step
                if t==start: 
                        Xout1=abs(xout1)    #choice between abs(x1) and x1 matching choice in training
                else:
                        Xout1=hstack((Xout1,abs(xout1)))


        Predlabel1=clf.predict(Xout1.T) 
        #Predlabel1=dot(Xout1.T,Wpred.T)

        sel=Predlabel1.mean(axis=0)
        sel=sel/sum(sel)
        most=max(sel)

        pred_class=np.where(sel==most)[0]
        #print('expected and predicted', mostest, choout)

        if pred_class==testval1:
                correct+=1

print ('Percentage Identified Correctly: ', correct/(i+1))
