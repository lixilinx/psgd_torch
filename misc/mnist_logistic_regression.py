import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd 

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
        self.W = torch.nn.Parameter(torch.zeros(28**2 + (28**2)**2,   10)) 
        self.b = torch.nn.Parameter(torch.zeros(10)) 
        
    def forward(self, x):
        x1 = x.view(-1, 28**2)
        x2 = torch.linalg.matmul(x1[:,:,None], x1[:,None,:]) 
        return torch.cat([x1, x2.view(-1, (28**2)**2)], 1).mm(self.W) + self.b
    
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

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()
num_iterations = 20

for resample in [False, True]:
    print("\nResample train data: {}\n".format(resample))

    """
        SGD baseline 
    """
    logistic.reset()
    opt = torch.optim.SGD(logistic.parameters(), lr=0.5)
    TrainLosses, best_test_err = [], 1.0
    total_time = 0.0
    for epoch in range(num_iterations):
        t0 = time.time()
        total_train_loss = 0.0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
                
            def xentropy():
                return train_loss(data, target)
            
            opt.zero_grad()
            loss = xentropy()
            loss.backward()
            opt.step()
            total_train_loss += loss.item()
        total_time += time.time() - t0 
                 
        TrainLosses.append(total_train_loss/len(train_loader))
        this_test_err = test_err( )
        if this_test_err < best_test_err:
            best_test_err = this_test_err
        opt.param_groups[0]['lr'] *= 0.01**(1/(num_iterations - 1))
        print('Epoch: {}; train loss: {}; SGD best test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )
    
    
    """
        L-BFGS baseline 
    """
    # LBFGS; unstable for lr=0.2; may diverge with lr=0.1; lr=0.05 may lead to poorer performance than SGD   
    logistic.reset()
    opt = torch.optim.LBFGS(logistic.parameters(), lr=0.1, max_iter=10, history_size=10)
    TrainLosses, best_test_err = [], 1.0
    total_time = 0.0
    for epoch in range(num_iterations):
        t0 = time.time()
        total_train_loss = 0.0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
                
            def xentropy():
                opt.zero_grad()
                xe = train_loss(data, target) 
                xe.backward()
                return xe
            
            loss = opt.step(xentropy)         
            total_train_loss += loss.item()
        total_time += time.time() - t0 
                 
        TrainLosses.append(total_train_loss/len(train_loader))
        this_test_err = test_err( )
        if this_test_err < best_test_err:
            best_test_err = this_test_err
        opt.param_groups[0]['lr'] *= 0.01**(1/(num_iterations))
        print('Epoch: {}; train loss: {}; LBFGS best test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )
    
    
    """
        PSGD 
    """
    logistic.reset()
    opt = psgd.LRA(logistic.parameters(), preconditioner_init_scale=None, lr_params=0.05, lr_preconditioner=0.1)
    TrainLosses, best_test_err = [], 1.0
    total_time = 0.0
    for epoch in range(num_iterations):
        t0 = time.time()
        total_train_loss = 0.0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
                
            def xentropy():
                return train_loss(data, target)
    
            loss = opt.step(xentropy)
            total_train_loss += loss.item()
        total_time += time.time() - t0 
                 
        TrainLosses.append(total_train_loss/len(train_loader))
        this_test_err = test_err()
        if this_test_err < best_test_err:
            best_test_err = this_test_err
        opt.lr_params *= 0.01**(1/(num_iterations - 1))
        print('Epoch: {}; train loss: {}; PSGD best test classification error rate: {}'.format(epoch+1, TrainLosses[-1], best_test_err))
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )


ax1.set_xlabel("Epochs")
ax1.set_ylabel("Regression loss")
ax1.tick_params(labelsize=7)
ax1.legend(
    [
        "SGD (w/o resample)",
        "L-BFGS (w/o resample)",
        "PSGD-LRA (w/o resample)",
        "SGD (w/ resample)",
        "L-BFGS (w/ resample)",
        "PSGD-LRA (w/ resample)",
    ],
    fontsize=7,
)
ax1.set_title("(a)")

ax2.set_xlabel("Wall time (s)")
ax2.tick_params(labelsize=7)
# ax2.set_ylabel("Fitting loss")
ax2.legend(
    [
        "SGD (w/o resample)",
        "L-BFGS (w/o resample)",
        "PSGD-LRA (w/o resample)",
        "SGD (w/ resample)",
        "L-BFGS (w/ resample)",
        "PSGD-LRA (w/ resample)",
    ],
    fontsize=7,
)
ax2.set_title("(b)")

plt.savefig("large_scale_logistic_regression.svg")
plt.show()