import matplotlib.pyplot as plt
import torch
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

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.W1 = torch.nn.Parameter(0.1*torch.randn(6,   1*5*5+1)) # CNN, (out_chs, in_chs*H*W + 1)
        self.W2 = torch.nn.Parameter(0.1*torch.randn(16,  6*5*5+1)) # CNN
        self.W3 = torch.nn.Parameter(0.1*torch.randn(16*4*4+1,120)) # FC
        self.W4 = torch.nn.Parameter(0.1*torch.randn(120+1,    84)) # FC
        self.W5 = torch.nn.Parameter(0.1*torch.randn(84+1,     10)) # FC
        
    def forward(self, x):
        x = F.conv2d(x, self.W1[:,:-1].view(6,1,5,5), bias=self.W1[:,-1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.conv2d(x, self.W2[:,:-1].view(16,6,5,5), bias=self.W2[:,-1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(x.view(-1, 16*4*4).mm(self.W3[:-1]) + self.W3[-1])
        x = F.relu(x.mm(self.W4[:-1]) + self.W4[-1])
        return x.mm(self.W5[:-1]) + self.W5[-1]
        
lenet5 = LeNet5()

@torch.jit.script
def train_loss(data, target):
    y = lenet5(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)     

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = lenet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target)            
    return num_errs.item()/len(test_loader.dataset)


Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in lenet5.parameters()]
lr = 0.1
grad_norm_clip_thr = 0.1*sum(W.numel() for W in lenet5.parameters())**0.5
TrainLosses, best_test_loss = [], 1.0
for epoch in range(10):
    for _, (data, target) in enumerate(train_loader):
        loss = train_loss(data, target)
        grads = torch.autograd.grad(loss, lenet5.parameters(), create_graph=True)
        vs = [torch.randn_like(W) for W in lenet5.parameters()]
        Hvs = torch.autograd.grad(grads, lenet5.parameters(), vs) 
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, Hvs)]
            pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            lr_adjust = min(grad_norm_clip_thr/grad_norm, 1.0)
            [W.subtract_(lr_adjust*lr*g) for (W, g) in zip(lenet5.parameters(), pre_grads)]                
            TrainLosses.append(loss.item())
    best_test_loss = min(best_test_loss, test_loss())
    lr *= (0.01)**(1/9)
    print('Epoch: {}; best test classification error rate: {}'.format(epoch+1, best_test_loss))
plt.plot(TrainLosses)