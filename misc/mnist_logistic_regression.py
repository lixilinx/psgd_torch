import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd 

print("""
      Compare PSGD with SGD/LBFGS on logistic regression. 
      LBFGS is the algorithm of choice for this type of problems. 
      When comparing the convergence speed, one should be aware that LBFGS may 
      take up to tens of iterations per step, while SGD/PSGD one iteration per step. 
      
      Typical results on MNIST
      
      Algorithm     Train regression loss       Test classification error rate 
      
      SGD           0.012                       2.45%
      SGD           0.012                       2.46%
      SGD           0.011                       2.30%
      SGD           0.015                       2.39%
      SGD           0.017                       2.43%
      SGD           0.012                       2.29%
      SGD           0.011                       2.25%
      SGD           0.011                       2.41%
      SGD           0.013                       2.28%
      SGD           0.012                       2.41%
      SGD           0.014                       2.40%
      
      LBFGS         0.012                       2.15%
      LBFGS         0.0056                      2.04%
      LBFGS         0.0085                      2.10%
      LBFGS         0.045 (before divergence)   2.41% (before divergence)
      LBFGS         0.0032                      1.90%      
      LBFGS         0.034 (before divergence)   2.33% (before divergence)
      LBFGS         0.28 (before divergence)    4.35% (before divergence)    
      LBFGS         0.0047                      2.02%    
      LBFGS         0.0036                      1.86%       
      LBFGS         0.0053                      2.01%
      LBFGS         0.012                       2.30%
      
      PSGD          0.00082                     2.06%
      PSGD          0.0011                      1.95%
      PSGD          0.0010                      1.84%
      PSGD          0.00072                     2.10%
      PSGD          0.00072                     2.04%
      PSGD          0.00078                     1.99%
      PSGD          0.00073                     2.04%
      PSGD          0.0010                      1.87%
      PSGD          0.00074                     2.02%
      PSGD          0.00085                     1.97%
      PSGD          0.00086                     1.91%
      """)

device = torch.device('cuda:0')

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=False)

class Logistic(torch.nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros((28*28)**2,   10)) 
        self.b = torch.nn.Parameter(torch.zeros(10)) 
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.linalg.matmul(x[:,:,None], x[:,None,:]) 
        return x.view(-1, (28*28)**2).mm(self.W) + self.b
    
    def reset(self):
        with torch.no_grad():
            self.W *= 0
            self.b *= 0
        
logistic = Logistic().to(device)

def train_loss(data, target):
    y = logistic(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)     

def test_err( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = logistic(data.to(device))
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target.to(device))            
    return num_errs.item()/len(test_loader.dataset)


# SGD
logistic.reset()
opt = torch.optim.SGD(logistic.parameters(), lr=0.5)
TrainLosses, best_test_err = [], 1.0
for epoch in range(20):
    total_train_loss = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        def xentropy():
            return train_loss(data, target)
        
        opt.zero_grad()
        loss = xentropy()
        loss.backward()
        opt.step()
        total_train_loss += loss.item()
             
    TrainLosses.append(total_train_loss/len(train_loader))
    this_test_err = test_err( )
    if this_test_err < best_test_err:
        best_test_err = this_test_err
    opt.param_groups[0]['lr'] *= 0.01**(1/19)
    print('Epoch: {}; train loss: {}; test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
plt.semilogy(TrainLosses, label='SGD')


# LBFGS; unstable for lr=0.2; may diverge with lr=0.1; lr=0.05 may lead to poorer performance than SGD   
logistic.reset()
opt = torch.optim.LBFGS(logistic.parameters(), lr=0.1, max_iter=10, history_size=10)
TrainLosses, best_test_err = [], 1.0
for epoch in range(20):
    total_train_loss = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        def xentropy():
            opt.zero_grad()
            xe = train_loss(data, target) 
            xe.backward()
            return xe
        
        loss = opt.step(xentropy)         
        total_train_loss += loss.item()
             
    TrainLosses.append(total_train_loss/len(train_loader))
    this_test_err = test_err( )
    if this_test_err < best_test_err:
        best_test_err = this_test_err
    opt.param_groups[0]['lr'] *= 0.01**(1/19)
    print('Epoch: {}; train loss: {}; test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
plt.semilogy(TrainLosses, label='LBFGS')


# PSGD 
logistic.reset()
opt = psgd.UVd(logistic.parameters(), lr_params=0.05, lr_preconditioner=0.1)
TrainLosses, best_test_err = [], 1.0
for epoch in range(20):
    total_train_loss = 0.0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data = torch.bernoulli(data) # we get slightly better test accuracy by enable this line 
        def xentropy():
            return train_loss(data, target)

        loss = opt.step(xentropy)
        total_train_loss += loss.item()
             
    TrainLosses.append(total_train_loss/len(train_loader))
    this_test_err = test_err()
    if this_test_err < best_test_err:
        best_test_err = this_test_err
    opt.lr_params *= 0.01**(1/19)
    print('Epoch: {}; train loss: {}; test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
plt.semilogy(TrainLosses, label='PSGD')
plt.xlabel('Epoch')
plt.ylabel('Train cross entropy')
plt.xlim(-1, 20)
plt.title('Note: LBFGS may take up to 10 iterations per step')
plt.legend()