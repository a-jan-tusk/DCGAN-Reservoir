#Using DCGANs for MNIST


from numpy import *
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

random.seed(42)


# Parameters
data_mean = 5
data_stddev = 1.5

# Model params
minibatch_size = 10   #100

lr=2e-4
optim_betas = (0.5, 0.999)
num_epochs = 100
d_steps = 1  #  Can train the discriminator at higher rate than generator
g_steps = 1

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', trazin=True, download=True, transform=transform),
batch_size=minibatch_size, shuffle=True)



##### Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



#Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


def extract(v):
    return v.data.storage().tolist()

def mean_std(d):
    return [np.mean(d), np.std(d)]


G = Generator()
D = Discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
loss = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=optim_betas)

for epoch in range(num_epochs):
    for x_, _ in train_loader:
         print ('hi')
         for d_index in range(d_steps):
             # 1. Train D on real+fake
             D.zero_grad()
             mini_batch = x_.size()[0]

             #  1A: Train D on real
             d_real_data = Variable(x_)
             d_real_decision = D(d_real_data).squeeze()
             d_real_error = loss(d_real_decision, Variable(torch.ones(mini_batch)))  # ones = true
             d_real_error.backward() # compute/store gradients, but don't change params

             #  1B: Train D on fake
             d_gen_input = noise(mini_batch).view(-1,100,1,1)
             d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
             d_fake_decision = D(d_fake_data).squeeze()
             d_fake_error = loss(d_fake_decision, Variable(torch.zeros(mini_batch)))  # zeros = fake
             d_fake_error.backward()
             d_optimizer.step()     # Only optimizes D's parameters based on stored gradients from backward()

         for g_index in range(g_steps):
             # 2. Train G on D's response (but DO NOT train D on these labels)
             G.zero_grad()

             gen_input = noise(mini_batch).view(-1,100,1,1)
             g_fake_data = G(gen_input)
             dg_fake_decision = D(g_fake_data).squeeze()
             g_error = loss(dg_fake_decision, Variable(torch.ones(mini_batch)))  # we want to fool, so pretend it's all true; train against ones

             g_error.backward()
             g_optimizer.step()  # Only optimizes G's parameters


    #display progress

    if epoch % 10 == 0:
        print("Epoch %s: D Realerror/fakeerror: %s/%s G error: %s (mean,std: Real: %s, Fake: %s) " % (epoch,
                                                            extract(d_real_error)[0],
                                                            extract(d_fake_error)[0],
                                                            extract(g_error)[0],
                                                            mean_std(extract(d_real_data)),
mean_std(extract(d_fake_data))))
