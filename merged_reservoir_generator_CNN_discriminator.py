import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

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
import numpy as np
import pickle, gzip
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
import json
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os



class generator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class = 10):
        super(generator, self).__init__()
        #self.fc1 = nn.Linear(input_size, 256)
        #self.fc2 = nn.Linear(self.fc1.out_features, 512)
        #self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(input_size, n_class)
        #self.x=torch.zeros([128, 784], dtype=torch.float)

    # forward method
    def forward(self, input):
        #x = F.leaky_relu(self.fc1(input), 0.2)
        #x = F.leaky_relu(self.fc2(x), 0.2)
        #x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.tanh(self.fc4(input))
        #x=torch.ones([128, 784], dtype=torch.float)
        #print '\nhere\n', x.size(), x.type()

        return x




class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))

        return x

WarmUpCycles = 2
rc_input_size = 100
rc_size=50
InitialSeedSize = 100

#gen_size=rc_size*int(InitialSeedSize/rc_input_size)
gen_size = 784

class our_RC:
    def __init__(self,size,inputsize):
        self.ReservoirSize = size
        self.InputSize = inputsize
        self.lam=1.0
        self.adj_res=(scipy.sparse.rand(self.ReservoirSize, self.ReservoirSize, density=0.1, format='coo', random_state=100).A-0.5)*2
        self.adj_res[self.adj_res==-1]=0
        self.adj_inp=(scipy.sparse.rand(self.ReservoirSize, self.InputSize +1, density=0.9, format='coo', random_state=100).A-0.5)*2
        self.adj_inp[self.adj_inp==-1]=0
        rho= math.sqrt(max(abs(linalg.eig(self.adj_res)[0])))
        sr=0.9
        self.adj_res *= sr/ rho

    def FeedForward(self,InputFeed):
        #print "input shape", InputFeed.shape
        InputFeedSize=InputFeed.shape[0]  #training size
        for i in range(InputFeedSize):  
            Image = InputFeed[i]
            Image=Image.reshape(self.InputSize,1)
            Image1=Image*0
            ReservoirState=np.zeros((self.ReservoirSize,1))
            
            for t in range(WarmUpCycles):
                
                if t==0:
                    ReservoirState=self.lam*np.tanh(np.dot(self.adj_inp,np.vstack((1,Image))) + np.dot(self.adj_res,ReservoirState)) #reservoir update step
                    AllReservoirState=abs(ReservoirState.reshape((1,self.ReservoirSize))) #choice between abs(ReservoirState) and ReservoirState
                else:
                    ReservoirState=self.lam*np.tanh(np.dot(self.adj_inp,np.vstack((1,Image1))) + np.dot(self.adj_res,ReservoirState)) #reservoir update step
                    AllReservoirState=np.hstack( (AllReservoirState, abs(ReservoirState.reshape((1,self.ReservoirSize)))) )   #choice between abs(ReservoirState) and ReservoirState
            #print "\nAllReservoirState.shape :", AllReservoirState.shape
            if i==0:
                    X=AllReservoirState
            else:
                    X=np.vstack((X,AllReservoirState))
        #print "\nXshape :",X.shape
        return X

    def FeedForwardColumn(self,InputFeed):
        InputFeedSize=InputFeed.shape[0]  #training size
        for i in range(InputFeedSize):  
            Image = InputFeed[i]
            NumberOfColumns = int(Image.shape[0]/self.InputSize)
            Image=Image.reshape(self.InputSize,int(Image.shape[0]/self.InputSize))
            ReservoirState=np.zeros((self.ReservoirSize,1))
            for t in range(NumberOfColumns):
                Column=np.zeros((self.InputSize,1))
                Column[:,0]=Image[:,t]
                ReservoirState=self.lam*np.tanh(np.dot(self.adj_inp,np.vstack((1,Column))) + np.dot(self.adj_res,ReservoirState)) #reservoir update step
                
                if t==0:
                    AllReservoirState=abs(ReservoirState.reshape((self.ReservoirSize))) #choice between abs(ReservoirState) and ReservoirState
                else:
                    AllReservoirState=np.hstack( (AllReservoirState, abs(ReservoirState.reshape((self.ReservoirSize)))) )   #choice between abs(ReservoirState) and ReservoirState
            if i==0:
                    X=AllReservoirState
            else:
                    X=np.vstack((X,AllReservoirState))
        return X



batch_size = 100


Resevoir = our_RC(rc_size,rc_input_size)

last_batch = 0
group = 0
def givemebatch(dataset):
    global last_batch
    max_minibatch_index = int(dataset.shape[0]/batch_size)
    minibatch = dataset[last_batch*batch_size:(last_batch+1)*batch_size]
    last_batch += 1
    last_batch = last_batch % max_minibatch_index
    return minibatch


#read input file generated by fpga

def readfile(filename):
    dataset = []
    with open(filename) as f:
        Lines = f.readlines()
    for line in Lines:
        dataset.append([int(I) for I in list(line.replace("\n","").replace(",",""))][:784])
    dataset = np.array(dataset)
    return dataset

HeidiResevoirStates = readfile("RPU_out.txt") ### the file that stroes queries to the resrvoir.

fixed_z_ = givemebatch(HeidiResevoirStates)
fixed_z_ = torch.tensor(fixed_z_, dtype=torch.float)




#fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = givemebatch(HeidiResevoirStates)
    z_ = torch.tensor(z_, dtype=torch.float)
    #z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()






def denorm(x):
	out = (x + 1) / 2
	return torch.clamp(out,0,1)


def show_images(images, filename):
    images = images.detach().numpy()
    w=100
    h=100
    fig=plt.figure(figsize=(8, 8))
    columns = 10
    rows = 10
    for i in range(1, columns*rows +1):
        img = images[i-1].reshape(28,28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.savefig("./"+filename)


# training parameters
#batch_size = 128
batch_size = 200
#lr = 0.0002
lr = 0.00001
train_epoch = 1000

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

train_loader_pretrain = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=10000, shuffle=True)

# network
G = generator(input_size=gen_size, n_class=28*28)
D = discriminator(input_size=28*28, n_class=1)
#G.cuda()
#D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
    os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
    os.mkdir('MNIST_GAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

image_labeled = [[] for _ in range(10)]


for i,(x_, y_) in enumerate(train_loader_pretrain):
    x_ = x_.numpy()
    y_ = y_.numpy()
    for j in range(y_.shape[0]):
        image_labeled[y_[j].item()].append(x_[j])
Average_Images = []
for i in range(10):
    temp = image_labeled[i]
    temp = np.array(temp)
    temp = np.mean(temp, axis=0)
    Average_Images.append(temp)

Average_Images = np.array(Average_Images)


Average_Images = torch.tensor(Average_Images,dtype=torch.float)


#show_images(denorm(Average_Images).permute(0,2,3,1))
    

distance = nn.MSELoss()

## pretraining 

for iterations in range(100):
    G.zero_grad()
    gen_in = torch.tensor(givemebatch(HeidiResevoirStates),dtype= torch.float)
    
    gen_out_label =[]
    for item in range(batch_size):
        min_loss = 2000000
        current_label = -1
        for label in range(10):
            d = distance(gen_in[item],Average_Images[label].view(784))
            
            if d < min_loss:
                min_loss = d
                current_label = label
        
        gen_out_label.append(Average_Images[current_label].view(784).numpy())
    

    #fake_label=torch.randn(100,784)


    gen_out_label_ = torch.tensor(gen_out_label,dtype=torch.float).view(-1,1,28,28)
    gen_out_label = torch.tensor(gen_out_label,dtype=torch.float)
    #print gen_out_label_.size()
    show_images(denorm(gen_out_label_).permute(0,2,3,1), "label.png")
    gen_out_label = Variable(torch.tensor(gen_out_label,dtype=torch.float))
    #gen_out_label = torch.tensor(gen_out_label,dtype=torch.float)
    #print "gen_out_label.size():",gen_out_label.size()
    gen_in = Variable(gen_in)
    #gen_in = gen_in
    G_result = G(gen_in)
    Gen_images = G_result.view(-1,1,28,28)
    Gen_error = MSE_loss(G_result,gen_out_label)
    #Gen_error = MSE_loss(G_result,fake_label)
    Gen_error.backward()
    G_optimizer.step()
    if iterations == 30:
        break
    show_images(denorm(Gen_images).permute(0,2,3,1),"Genimage.png")
   




for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    
    for x_, _ in train_loader:
        # train discriminator D
        if epoch<50:
            i_dis=5
        elif epoch>50 and epoch<100:
            i_dis=2
        else:
            i_dis=1

        for i in range(i_dis):
            D.zero_grad()

            x_ = x_.view(-1, 28 * 28)

            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            #x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            D_result = D(x_)
            D_real_loss = BCE_loss(D_result, y_real_)
            D_real_score = D_result

            z_ = givemebatch(HeidiResevoirStates)
            z_ = torch.tensor(z_, dtype=torch.float)
            #z_ = Variable(z_.cuda())
            G_result = G(z_)

            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

        D_losses.append(D_train_loss.data[0])
        
        # train generator G        
        if epoch<50:
            i_gen=2
        elif epoch>50 and epoch<100:
            i_gen=5
        else:
            i_gen=1

        for i in range(i_gen):
            G.zero_grad()

            #z_ = torch.randn((mini_batch, 100))
            z_ = givemebatch(HeidiResevoirStates)
            z_ = torch.tensor(z_, dtype=torch.float)
            y_ = torch.ones(mini_batch)

            #z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())
            G_result = G(z_)
            D_result = D(G_result)
            G_train_loss = BCE_loss(D_result, y_)
            G_train_loss.backward()
            G_optimizer.step()
        G_losses.append(G_train_loss.data[0])

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


print("Training finish!... save training results")
torch.save(G.state_dict(), "MNIST_GAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_GAN_results/discriminator_param.pkl")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)
