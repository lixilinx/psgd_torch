import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=True)

"""input image size for the original LeNet5 is 32x32, here is 28x28"""
W1 = torch.tensor(0.1*torch.randn(1*5*5+1,  6), requires_grad=True)
W2 = torch.tensor(0.1*torch.randn(6*5*5+1,  16), requires_grad=True)
W3 = torch.tensor(0.1*torch.randn(16*4*4+1, 120), requires_grad=True)#so here is 4x4, not 5x5
W4 = torch.tensor(0.1*torch.randn(120+1,    84), requires_grad=True)
W5 = torch.tensor(0.1*torch.randn(84+1,     10), requires_grad=True)
Ws = [W1, W2, W3, W4, W5]

def LeNet5(x): 
    x = F.conv2d(x, W1[:-1].view(6,1,5,5), bias=W1[-1])
    x = torch.tanh(F.max_pool2d(x, 2))
    x = F.conv2d(x, W2[:-1].view(16,6,5,5), bias=W2[-1])
    x = torch.tanh(F.max_pool2d(x, 2))
    x = torch.tanh(x.view(-1, 16*4*4).mm(W3[:-1]) + W3[-1])
    x = torch.tanh(x.mm(W4[:-1]) + W4[-1])
    y = x.mm(W5[:-1]) + W5[-1]
    return y

def train_loss(data, target):
    y = LeNet5(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    for W in Ws:
        loss += 0.0002*torch.sum(W*W)
        
    return loss

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = LeNet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target)
            
    return num_errs.item()/len(test_loader.dataset)

Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in Ws]
step_size = 0.1
grad_norm_clip_thr = 1e10
TrainLoss, TestLoss = [], []
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_loss(data, target)
        
        grads = grad(loss, Ws, create_graph=True)
        TrainLoss.append(loss.item())
        if batch_idx%10 == 0:
            print('Epoch: {}; batch: {}; train loss: {}'.format(epoch, batch_idx, TrainLoss[-1]))
        
        v = [torch.randn(W.shape) for W in Ws]
        Hv = grad(grads, Ws, v)#let Hv=grads if using whitened gradients   
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
            pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= step_adjust*step_size*pre_grads[i]
                
    TestLoss.append(test_loss())
    step_size = (0.1**0.1)*step_size
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))