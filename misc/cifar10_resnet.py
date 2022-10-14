import sys
import math
import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

parser = argparse.ArgumentParser()
parser.add_argument("device",                           help="for example, cuda:0")
parser.add_argument("optimizer",                        help="choices are SGD, PSGD_XMat and PSGD_UVd")
parser.add_argument("lr_scheduler",                     help="choices are stage and cos")
parser.add_argument("shortcut_connection", type=int,    help="choices are 0 and 1")

args = parser.parse_args()
device = torch.device(args.device)
optimizer = args.optimizer
lr_scheduler = args.lr_scheduler
shortcut_connection = bool(args.shortcut_connection)
print("Device: \t\t\t{}".format(device))
print("Optimizer: \t\t\t{}".format(optimizer))
print("Learning rate schedular:\t{}".format(lr_scheduler))
print("With short connections: \t{}".format(shortcut_connection))

if optimizer == 'SGD':
    lr0 = 1.0   # 0.1 -> 1.0 when momentum factor = 0.9 as momentum in PSGD is the moving average of gradient 
    decay = 5e-4
else: # PSGD_XMat or PSGD_UVd    
    lr0 = 2e-2
    if shortcut_connection:
        decay = 2e-2
    else:
        decay = 1e-2
    
#print(
    """
      The code is adapted from the Adabelief implementation at
          https://github.com/juntang-zhuang/Adabelief-Optimizer
          
      We test three optimizers: SGD, PSGD_XMat and PSGD_UVd (PSGD with the two preconditioners).
      Two learning rate schedulers: three-stage annealing and cosine annealling. 
      Two variations: with and without the shortcut connections. 
      
      Minor change 1: replacing ReLU(x) with 0.51*x + 0.49*sqrt(x^2 + eps^2) helps to round  
      the derivatives around 0 for better numerical behaviors. 
      Minor change 2: replacing L2 regularization 0.5*decay*x^2 with decay*rand()*x^2 seems help.
      
      Test accuracies of a few runs with resnet18:
          
      WITH SHORTCUT CONNECTIONS
          
        algorithm   lr_scheduler    lr0         weight_decay    test_accuracy
    
        SGD         stage           1e0         5e-4            95.13
        SGD         stage           1e0         5e-4            95.02
        SGD         stage           1e0         5e-4            95.35
        SGD         stage           1e0         5e-4            94.96
        SGD         stage           1e0         5e-4            95.02
        SGD         stage           1e0         5e-4            95.04
        SGD         stage           1e0         5e-4            94.83
                                                                        95.05+-0.15
        
        PSGD(UVd)   stage           2e-2        2e-2            95.45      
        PSGD(UVd)   stage           2e-2        2e-2            95.35
        PSGD(UVd)   stage           2e-2        2e-2            95.57  
        PSGD(UVd)   stage           2e-2        2e-2            95.45
        PSGD(UVd)   stage           2e-2        2e-2            95.48
        PSGD(UVd)   stage           2e-2        2e-2            95.55
        PSGD(Xmat)  stage           2e-2        2e-2            95.57 
        PSGD(Xmat)  stage           2e-2        2e-2            95.49 
        PSGD(Xmat)  stage           2e-2        2e-2            95.38
        PSGD(Xmat)  stage           2e-2        2e-2            95.50
        PSGD(UVd)   stage           3e-2        2e-2            95.46
                                                                        95.48+-0.07
        
        SGD         cos             1e0         5e-4            95.24
        SGD         cos             1e0         5e-4            95.65
        SGD         cos             1e0         5e-4            95.25
        SGD         cos             1e0         5e-4            95.50
        SGD         cos             1e0         5e-4            95.58
        SGD         cos             1e0         5e-4            95.59
        SGD         cos             1e0         5e-4            95.38
        SGD         cos             1e0         5e-4            95.38
        SGD         cos             1e0         5e-4            95.53 
        SGD         cos             1e0         5e-4            95.64
                                                                        95.47+-0.14
        
        PSGD(UVd)   cos             2e-2        2e-2            95.55
        PSGD(UVd)   cos             2e-2        2e-2            95.44
        PSGD(UVd)   cos             2e-2        2e-2            95.56
        PSGD(UVd)   cos             2e-2        2e-2            95.50
        PSGD(UVd)   cos             2e-2        2e-2            95.45
        PSGD(XMat)  cos             2e-2        2e-2            95.45
        PSGD(XMat)  cos             2e-2        2e-2            95.42
        PSGD(XMat)  cos             2e-2        2e-2            95.69 
        PSGD(XMat)  cos             2e-2        2e-2            95.32
        PSGD(UVd)   cos             5e-2        2e-2            95.48
        PSGD(UVd)   cos             5e-2        2e-2            95.46
        PSGD(UVd)   cos             4e-2        2e-2            95.55
        PSGD(UVd)   cos             3e-2        2e-2            95.57
        PSGD(UVd)   cos             2e-2        1e-2            95.45
        PSGD(UVd)   cos             2e-2        3e-2            95.51
                                                                        95.49+-0.08
                                                                        
      REMOVE SHORTCUT CONNECTIONS
     
        SGD         cos             1e0         5e-4            94.92
        SGD         cos             1e0         5e-4            95.21
        SGD         cos             1e0         5e-4            94.80
        SGD         cos             1e0         5e-4            95.07
        SGD         cos             1e0         5e-4            94.93
        SGD         cos             1e0         5e-4            94.87
        SGD         cos             1e0         5e-4            94.88
        SGD         cos             1e0         5e-4            95.16
        SGD         cos             1e0         5e-4            94.92
        SGD         cos             1e0         5e-4            95.02
        SGD         cos             1e0         5e-4            94.82
        SGD         cos             1e0         5e-4            95.12
                                                                        94.98+-0.13
                                                                        
        PSGD(UVd)   cos             2e-2        1e-2            95.53
        PSGD(UVd)   cos             2e-2        1e-2            95.38
        PSGD(UVd)   cos             2e-2        1e-2            95.31
        PSGD(UVd)   cos             2e-2        1e-2            95.46
        PSGD(UVd)   cos             2e-2        1e-2            95.21
        PSGD(UVd)   cos             2e-2        1e-2            95.54
        PSGD(UVd)   cos             2e-2        1e-2            95.30
        PSGD(UVd)   cos             2e-2        1e-2            95.39
        PSGD(XMat)  cos             2e-2        1e-2            95.44
        PSGD(XMat)  cos             2e-2        1e-2            95.33
        PSGD(XMat)  cos             2e-2        1e-2            95.39
        PSGD(XMat)  cos             2e-2        1e-2            95.44
                                                                        95.39+-0.09
            
      PSGD also is less sensitive to the way of weight decaying 
      (coupled here, and two more decoupled ways). 
      """
#)

def soft_lrelu(x):
    # Reducing to ReLU when a=0.5 and e=0
    # Here, we set a-->0.5 from left and e-->0 from right,
    # where adding eps is to make the derivatives have better rounding behavior around 0. 
    a = 0.49 
    e = torch.finfo(torch.float32).eps
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

        if shortcut_connection:
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
        if shortcut_connection:
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

        if shortcut_connection:
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
        if shortcut_connection:
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
    opt = psgd.XMat(
        net.parameters(),
        lr_params = lr0, # note that momentum in PSGD is the moving average of gradient
        momentum = 0.9,  # so lr 0.1 becomes 1 when momentum factor is 0.9
        preconditioner_update_probability = 0.0, # PSGD reduces to SGD when P = eye()
    )
elif optimizer == 'PSGD_XMat':
    # PSGD with X-shape matrix preconditioner
    opt = psgd.XMat(
        net.parameters(),
        lr_params = lr0,
        momentum = 0.9,
        preconditioner_update_probability = 0.1,
    )
else:
    # PSGD with low rank approximation preconditioner
    opt = psgd.UVd(
        net.parameters(),
        lr_params = lr0,
        momentum = 0.9,
        preconditioner_update_probability = 0.1,
    )

if shortcut_connection:
    train_loader, test_loader = build_dataset(128)
else:
    train_loader, test_loader = build_dataset(64)

criterion = nn.CrossEntropyLoss()
num_epoch = 200
train_accs = []
test_accs = []
for epoch in range(num_epoch):
    if lr_scheduler == 'cos':
        opt.lr_params = lr0*(1 + math.cos(math.pi*epoch/num_epoch))/2
    else:
        # schedule the learning rate
        if epoch == int(num_epoch * 0.7):
            opt.lr_params *= 0.1
        if epoch == int(num_epoch * 0.9):
            opt.lr_params *= 0.1
    
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
            loss = criterion(outputs, targets) + sum(
                [torch.sum(decay * torch.rand_like(param) * param * param) for param in net.parameters()]
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

    train_accs.append(train_accuracy)
    test_accs.append(test_accuracy)
print("train_accuracy: {}".format(train_accs))
print("test_accuracy: {}".format(test_accs))