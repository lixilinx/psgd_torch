import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=False)

LeNet5_vars = [0.1*torch.randn(1*5*5+1,  6),
               0.1*torch.randn(6*5*5+1,  16),
               0.1*torch.randn(16*4*4+1, 120),
               0.1*torch.randn(120+1,    84),
               0.1*torch.randn(84+1,     10),]
[W.requires_grad_(True) for W in LeNet5_vars]

def LeNet5(x): 
    W1, W2, W3, W4, W5 = LeNet5_vars
    x = F.conv2d(x, W1[:-1].view(6,1,5,5), bias=W1[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.conv2d(x, W2[:-1].view(16,6,5,5), bias=W2[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.relu(x.view(-1, 16*4*4).mm(W3[:-1]) + W3[-1])
    x = F.relu(x.mm(W4[:-1]) + W4[-1])
    return x.mm(W5[:-1]) + W5[-1]

def train_loss(data, target):
    y = LeNet5(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)     

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = LeNet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target)            
    return num_errs.item()/len(test_loader.dataset)


Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in LeNet5_vars]
lr = 0.1
grad_norm_clip_thr = 0.1*sum(W.numel() for W in LeNet5_vars)**0.5
TrainLosses, best_test_loss = [], 1.0
for epoch in range(10):
    for _, (data, target) in enumerate(train_loader):
        loss = train_loss(data, target)
        grads = grad(loss, LeNet5_vars, create_graph=True)
        vs = [torch.randn(W.shape) for W in LeNet5_vars]
        Hvs = grad(grads, LeNet5_vars, vs) 
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, Hvs)]
            pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            lr_adjust = min(grad_norm_clip_thr/grad_norm, 1.0)
            [W.subtract_(lr_adjust*lr*g) for (W, g) in zip(LeNet5_vars, pre_grads)]                
            TrainLosses.append(loss.item())
    best_test_loss = min(best_test_loss, test_loss())
    lr *= (0.01)**(1/9)
    print('Epoch: {}; best test classification error rate: {}'.format(epoch+1, best_test_loss))
plt.plot(TrainLosses)
