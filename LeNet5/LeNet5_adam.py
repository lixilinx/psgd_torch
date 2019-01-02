import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=True)

"""input image size for the original LeNet5 is 32x32, here is 28x28"""
W1 = torch.tensor(0.1*torch.randn(1*5*5+1,  6), requires_grad=True)
W2 = torch.tensor(0.1*torch.randn(6*5*5+1,  16), requires_grad=True)
W3 = torch.tensor(0.1*torch.randn(16*4*4+1, 120), requires_grad=True)#here is 4x4, not 5x5
W4 = torch.tensor(0.1*torch.randn(120+1,    84), requires_grad=True)
W5 = torch.tensor(0.1*torch.randn(84+1,     10), requires_grad=True)
Ws = [W1, W2, W3, W4, W5]

def LeNet5(x): 
    x = F.conv2d(x, W1[:-1].view(6,1,5,5), bias=W1[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.conv2d(x, W2[:-1].view(16,6,5,5), bias=W2[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.relu(x.view(-1, 16*4*4).mm(W3[:-1]) + W3[-1])
    x = F.relu(x.mm(W4[:-1]) + W4[-1])
    y = x.mm(W5[:-1]) + W5[-1]
    return y

def train_loss(data, target):
    y = LeNet5(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    return loss

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = LeNet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target)
    return num_errs.item()/len(test_loader.dataset)

m0 = [torch.zeros(W.shape) for W in Ws]
v0 = [torch.zeros(W.shape) for W in Ws]
step_size = 0.01
cnt = 0
TrainLoss, TestLoss = [], []
for epoch in range(10):
    trainloss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_loss(data, target)
        
        grads = grad(loss, Ws)#, create_graph=True)
        trainloss += loss.item()
        
        with torch.no_grad():
            lmbd = min(cnt/(cnt+1), 0.9)
            m0 = [lmbd*old + (1.0-lmbd)*new for (old, new) in zip(m0, grads)]
            lmbd = min(cnt/(cnt+1), 0.999)
            v0 = [lmbd*old + (1.0-lmbd)*new*new for (old, new) in zip(v0, grads)]
            for i in range(len(Ws)):
                Ws[i] -= step_size*m0[i]/torch.sqrt(v0[i] + 1e-8)
                
            cnt = cnt + 1
            
    TrainLoss.append(trainloss/len(train_loader.dataset))
    TestLoss.append(test_loss())
    step_size = 0.01**(1/9)*step_size
    print('Epoch: {}; train loss: {}; best test loss: {}'.format(epoch, TrainLoss[-1], min(TestLoss)))

plt.subplot(2,1,1)
plt.semilogy(range(1,11), TrainLoss, '-c', linewidth=0.2)
plt.xlabel('Epochs')
plt.ylabel('Train cross entropy loss')
plt.subplot(2,1,2)
plt.semilogy(range(1,11), TestLoss, '-c', linewidth=0.2)
plt.xlabel('Epochs')
plt.ylabel('Test classification error rate')