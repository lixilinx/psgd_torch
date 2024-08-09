import copy
import sys
import time

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cuda")


class AffineConv2d(torch.nn.Module):
    """
    Let's wrap function
        torch.nn.functional.conv2d
    as a class. The affine transform is
        [vectorized(image patch), 1] @ W
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=True
    ):
        super(AffineConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.has_bias = has_bias
        self.out_in_height_width = (out_channels, in_channels, kernel_size, kernel_size)

        std = (in_channels * kernel_size**2) ** (-0.5)
        w = torch.empty(out_channels, in_channels * kernel_size**2).normal_(std=std)
        if has_bias:
            b = torch.zeros(out_channels, 1)
            self.weight = torch.nn.Parameter(torch.cat([w, b], dim=1))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return F.conv2d(
                x,
                self.weight[:, :-1].view(self.out_in_height_width),
                bias=self.weight[:, -1],
                stride=self.stride,
                padding=self.padding,
            )
        else:
            return F.conv2d(
                x,
                self.weight.view(self.out_in_height_width),
                stride=self.stride,
                padding=self.padding,
            )


class AffineLinear(torch.nn.Module):
    """
    A linear layer clearly is an affine transform
    """

    def __init__(self, in_features, out_features, has_bias=True):
        super(AffineLinear, self).__init__()
        self.has_bias = has_bias
        w = torch.empty(in_features, out_features).normal_(std=in_features ** (-0.5))
        if has_bias:
            b = torch.zeros(1, out_features)
            self.weight = torch.nn.Parameter(torch.cat([w, b]))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return x @ self.weight[:-1] + self.weight[-1]
        else:
            return x @ self.weight


"""
Now let's test our affine components
"""
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=64,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=False, transform=transforms.Compose([transforms.ToTensor()])
    ),
    batch_size=1000,
    shuffle=False,
)


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = AffineConv2d(1, 6, 5)
        self.conv2 = AffineConv2d(6, 16, 5)
        self.linear1 = AffineLinear(16 * 4 * 4, 120)
        self.linear2 = AffineLinear(120, 84)
        self.linear3 = AffineLinear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(self.linear1(x.view(-1, 16 * 4 * 4)))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


def train_loss(data, target):
    y = lenet5(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)


def test_loss():
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            y = lenet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred != target)
    return num_errs.item() / len(test_loader.dataset)


# lenet5 with affine transform components
net = LeNet5()

# plt.figure(figsize=[9, 4])
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()

num_iterations = 10

for resample in [False, True]:
    print("\nResample train data: {}\n".format(resample))
    """
        SGD baseline
    """
    lenet5 = copy.deepcopy(net).to(device)
    
    TrainLosses, best_test_loss = [], 1.0
    lr = 0.2
    total_time = 0.0
    for epoch in range(num_iterations):
        total_loss = 0.0
        t0 = time.time()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
    
            def closure():
                return train_loss(data, target)
    
            loss = closure()
            total_loss += loss.item()
            grads = torch.autograd.grad(loss, lenet5.parameters())
            with torch.no_grad():
                [W.subtract_(lr * g) for (W, g) in zip(lenet5.parameters(), grads)]
    
        total_time += time.time() - t0
        TrainLosses.append(total_loss/len(train_loader))
    
        best_test_loss = min(best_test_loss, test_loss())
        lr *= (0.1) ** (1 / (num_iterations - 1))
        print(
            f"Epoch: {epoch + 1}; SGD best test classification error rate: {best_test_loss}"
        )
    
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )
    
    
    """
        Baseline with approximate closed-form solution for preconditioners (basically Shampoo)
    """
    lenet5 = copy.deepcopy(net).to(device)
    Rs = [
        [
            1e0 * torch.eye(W.shape[0], device=W.device),
            1e0 * torch.eye(W.shape[1], device=W.device),
        ]
        for W in lenet5.parameters()
    ]
    
    
    def RtoP(R):
        # return P = R^(-0.25)
        L, U = torch.linalg.eigh(R)
        P = U @ torch.diag(L ** (-0.25)) @ U.t()
        return P
    
    
    TrainLosses, best_test_loss = [], 1.0
    lr, grad_norm_clip_thr = 1.0, 10.0
    total_time = 0.0
    for epoch in range(num_iterations):
        total_loss = 0.0
        t0 = time.time()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
    
            def closure():
                return train_loss(data, target)
    
            loss = closure()
            total_loss += loss.item()
            grads = torch.autograd.grad(loss, lenet5.parameters())
            [
                (R[0].mul_(0.999).add_(g @ g.t()), R[1].mul_(0.999).add_(g.t() @ g))
                for (R, g) in zip(Rs, grads)
            ]
            pre_grads = [RtoP(R[0]) @ g @ RtoP(R[1]) for (R, g) in zip(Rs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g * g) for g in pre_grads]))
            lr_adjust = min(grad_norm_clip_thr / grad_norm, 1.0)
            with torch.no_grad():
                [
                    W.subtract_(lr_adjust * lr * g)
                    for (W, g) in zip(lenet5.parameters(), pre_grads)
                ]
        total_time += time.time() - t0
        TrainLosses.append(total_loss/len(train_loader))
    
        best_test_loss = min(best_test_loss, test_loss())
        lr *= (0.1) ** (1 / (num_iterations - 1))
        print(f"Epoch: {epoch + 1}; (basically Shampoo) best test classification error rate: {best_test_loss}")
    
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )
    
    
    """
        PSGD with Affine preconditioner 
    """
    lenet5 = copy.deepcopy(net).to(device)
    opt = psgd.Affine(
        lenet5.parameters(),
        preconditioner_init_scale=1.0,
        lr_params=0.1,
        lr_preconditioner=0.1,
        grad_clip_max_norm=10.0,
    )
    # opt = psgd.LRA(lenet5.parameters(), preconditioner_init_scale=None, lr_params=0.1, lr_preconditioner=0.1, grad_clip_max_norm=10.0)
    
    TrainLosses, best_test_loss = [], 1.0
    total_time = 0.0
    for epoch in range(num_iterations):
        total_loss = 0.0
        t0 = time.time()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if resample:
                data = torch.bernoulli(data)
    
            def closure():
                xentropy = train_loss(data, target) 
                # we add L2 term, eps*sum(p*p), to avoid hessian underflow
                L2 = 1e-9*sum([torch.sum(torch.rand_like(p) * p * p) for p in opt._params_with_grad])
                return (xentropy + L2, xentropy)
    
            _, loss = opt.step(closure)
            total_loss += loss.item()
        total_time += time.time() - t0
        TrainLosses.append(total_loss/len(train_loader))
    
        best_test_loss = min(best_test_loss, test_loss())
        opt.lr_params *= (0.01) ** (1 / (num_iterations - 1))
        print(
            f"Epoch: {epoch + 1}; PSGD best test classification error rate: {best_test_loss}"
        )
    
    ax1.semilogy(torch.arange(1, num_iterations + 1).cpu(), TrainLosses)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        TrainLosses,
    )


ax1.set_xlabel("Epochs")
ax1.set_ylabel("Train loss")
ax1.tick_params(labelsize=8)
ax1.legend(
    [
        "SGD (w/o resample)",
        "Shampoo (w/o resample)",
        "PSGD-Affine (w/o resample)",
        "SGD (w/ resample)",
        "Shampoo (w/ resample)",
        "PSGD-Affine (w/ resample)",
    ],
    fontsize=7,
)
ax1.set_title("(a)")

ax2.set_xlabel("Wall time (s)")
ax2.tick_params(labelsize=8)
# ax2.set_ylabel("Fitting loss")
ax2.legend(
    [
        "SGD (w/o resample)",
        "Shampoo (w/o resample)",
        "PSGD-Affine (w/o resample)",
        "SGD (w/ resample)",
        "Shampoo (w/ resample)",
        "PSGD-Affine (w/ resample)",
    ],
    fontsize=7,
)
ax2.set_title("(b)")

plt.savefig("sgd_vs_shampoo_vs_psgd.svg")
plt.show()
