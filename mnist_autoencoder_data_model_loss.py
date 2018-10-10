"""MNIST autoencoder example 
I followed Yaroslav Bulatov's setup, see:
https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
"""
import numpy as np
import tensorflow as tf#just for loading MNIST data
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# experiment settings
train_batch_size = 1000
dim1, dim2, dim3, dim4 = 1024, 1024, 1024, 196
dsize = 10000#only use 10000 samples for training and testing/validation, respetively

mnist = tf.contrib.learn.datasets.load_dataset('mnist')
train_data = torch.tensor(mnist.train.images[:dsize], dtype=torch.float).to(device)
test_data = torch.tensor(mnist.train.images[dsize+1:2*dsize], dtype=torch.float).to(device)

def get_batches():#for training
    rp = np.random.permutation(train_data.shape[0])
    x = train_data[rp[0:train_batch_size]]
    return x, x

# copied the same weight initialization method for fair comparison
def ng_init(s1, s2): # uniform weight init from Ng UFLDL
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  flat = np.random.random(s1*s2)*2*r-r
  return flat.reshape([s1, s2])

# Model Coefficients
dim0 = train_data.shape[1]
W1 = torch.tensor(ng_init(dim0+1, dim1), dtype=torch.float, requires_grad=True).to(device)
W2 = torch.tensor(ng_init(dim1+1, dim2), dtype=torch.float, requires_grad=True).to(device)
W3 = torch.tensor(ng_init(dim2+1, dim3), dtype=torch.float, requires_grad=True).to(device)
W4 = torch.tensor(ng_init(dim3+1, dim4), dtype=torch.float, requires_grad=True).to(device)
W5 = torch.tensor(ng_init(dim4+1, dim3), dtype=torch.float, requires_grad=True).to(device)
W6 = torch.tensor(ng_init(dim3+1, dim2), dtype=torch.float, requires_grad=True).to(device)
W7 = torch.tensor(ng_init(dim2+1, dim1), dtype=torch.float, requires_grad=True).to(device)
W8 = torch.tensor(ng_init(dim1+1, dim0), dtype=torch.float, requires_grad=True).to(device)
Ws = [W1, W2, W3, W4, W5, W6, W7, W8]   # put all trainable coefficients in this list

# Model Definition
def model(Ws, x0):
    W1, W2, W3, W4, W5, W6, W7, W8 = Ws
    num_samples = x0.shape[0]
    ones = torch.ones(num_samples, 1).to(device)
    x1 = torch.sigmoid( torch.cat([x0, ones], 1).mm(W1) )
    x2 = torch.sigmoid( torch.cat([x1, ones], 1).mm(W2) )
    x3 = torch.sigmoid( torch.cat([x2, ones], 1).mm(W3) )
    x4 = torch.sigmoid( torch.cat([x3, ones], 1).mm(W4) )
    x5 = torch.sigmoid( torch.cat([x4, ones], 1).mm(W5) )
    x6 = torch.sigmoid( torch.cat([x5, ones], 1).mm(W6) )
    x7 = torch.sigmoid( torch.cat([x6, ones], 1).mm(W7) )
    x8 = torch.sigmoid( torch.cat([x7, ones], 1).mm(W8) )
    return x8
    
# training criterion
def train_criterion(Ws, x, y):
    num_samples = x.shape[0]
    y_hat = model(Ws, x)
    return torch.sum((y - y_hat)**2)/2/num_samples

# testing criterion
def test_criterion(Ws):
    num_samples = test_data.shape[0]
    y_hat = model(Ws, test_data)
    return torch.sum((test_data - y_hat)**2)/2/num_samples