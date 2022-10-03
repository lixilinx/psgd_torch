import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cuda:0")
# optimizer = 'SGD'
# optimizer = 'PSGD XMat'
optimizer = 'PSGD UVd'

print(
    """
      The code is adapted from the Adabelief implementation at
          https://github.com/juntang-zhuang/Adabelief-Optimizer

      One main difference is that I anneal the learning rate twice.
      This change is not helpful to PSGD, but benefits the SGD baseline a lot.
      It increases the test accuracy of SGD by about 1%.
      
      Replacing ReLU with (x+sqrt(x^2+eps))/2 helps to deliver the correct rounding behavior 
      for derivatives around 0, where eps is the machine precision. 
      This helps PSGD to give a finer preconditioner estimation.   
      """
)

def soft_lrelu(x):
    # Reducing to ReLU when a=0.5 and e=0
    # Here, we set a-->0.5 from left and e-->0 from right,
    # where adding eps is to make the derivatives have the corrrect rounding behavior around 0. 
    a = (1 - 0.01)/2 
    e = torch.finfo(torch.float32).eps**0.5
    return (1-a)*x + a*torch.sqrt(x*x + e*e) - a*e

def build_dataset(batchsize):
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batchsize, shuffle=False
    )

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = soft_lrelu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = soft_lrelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = soft_lrelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total

    return accuracy


net = ResNet18().to(device)

if optimizer == 'SGD':
    # SGD baseline
    L2 = 2.5e-4 # 2*L2 is the weight decay
    opt = psgd.XMat(
        net.parameters(),
        lr_params = 1.0, # note that momentum in PSGD is the moving average of gradient
        momentum = 0.9,  # so lr 0.1 becomes 1 when momentum factor is 0.9
        preconditioner_update_probability = 0.0, # PSGD reduces to SGD when P = eye()
    )
elif optimizer == 'PSGD XMat':
    # PSGD with X-shape matrix preconditioner
    L2 = 1e-2 # 2*L2 is the weight decay
    opt = psgd.XMat(
        net.parameters(),
        momentum = 0.9,
        preconditioner_update_probability = 0.1,
    )
else:
    # PSGD with low rank approximation preconditioner
    L2 = 1e-2  # 2*L2 is the weight decay
    opt = psgd.UVd(
        net.parameters(),
        momentum = 0.9,
        preconditioner_update_probability = 0.1,
    )

train_loader, test_loader = build_dataset(128)

criterion = nn.CrossEntropyLoss()
num_epoch = 200
train_accs = []
test_accs = []
for epoch in range(num_epoch):
    net.train()  # do not forget it as there is BN
    total = 0
    train_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        def closure():
            """
            Weight decaying is explicitly realized by adding L2 regularization to the loss
            """
            outputs = net(inputs)
            loss = criterion(outputs, targets) + L2 * sum(
                [torch.sum(param * param) for param in net.parameters()]
            )
            return [loss, outputs]

        loss, outputs = opt.step(closure)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = 100.0 * correct / total
    test_accuracy = test(net, device, test_loader, criterion)
    print(
        "epoch: {}; train loss: {}; train accuracy: {}; test accuracy: {}".format(
            epoch + 1, train_loss, train_accuracy, test_accuracy
        )
    )

    # schedule the learning rate
    if epoch + 1 == int(num_epoch * 0.7):
        opt.lr_params *= 0.1
    if epoch + 1 == int(num_epoch * 0.9):
        opt.lr_params *= 0.1

    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)
print("train_accuracy: {}".format(train_accs))
print("test_accuracy: {}".format(test_accs))
