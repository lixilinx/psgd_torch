import sys

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
        w = torch.empty(out_channels, in_channels * kernel_size ** 2).normal_(std=std)
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

# lenet5 with affine transform components 
lenet5 = LeNet5().to(device)


@torch.jit.script
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


opt = psgd.Affine(lenet5.parameters(), lr_params=0.1, grad_clip_max_norm=20.0)

TrainLosses, best_test_loss = [], 1.0
for epoch in range(10):
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def closure():
            return train_loss(data, target) + 1e-6*sum([torch.sum(p*p) for p in opt._params_with_grad])

        TrainLosses.append(opt.step(closure).item())

    best_test_loss = min(best_test_loss, test_loss())
    opt.lr_params *= (0.01) ** (1 / 9)
    print(f"Epoch: {epoch + 1}; best test classification error rate: {best_test_loss}")