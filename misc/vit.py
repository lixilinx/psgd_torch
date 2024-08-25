"""
ViT demo. 
The ViT model is from
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import copy

import sys
import time

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cuda")

"""
Prepare the dataset
"""
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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

"""
the ViT model is from
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# a tiny vit model
Net = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=4,
    heads=8,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1,
)


"""
Test accuracy
"""


def test(net, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return accuracy


"""
Now we compare Adam(W) (the default optimizer for transformer) and PSGD.
We align their settings, and the only difference is their preconditioners.  
"""
num_epochs = 100
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()

"""
Adam 
"""
net = copy.deepcopy(Net).to(device)
lr0 = 1e-3
opt = torch.optim.Adam(net.parameters(), lr=lr0)  # will aneal lr to lr0/num_epochs

TrainLoss = []
TestAcc = []
t0 = time.time()
for epoch in range(num_epochs):
    """train"""
    net.train()
    for _, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        def closure():
            outputs = net(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            return loss

        opt.zero_grad()
        loss = closure()
        loss.backward()
        opt.step()
        TrainLoss.append(loss.item())

    """test"""
    net.eval()
    test_acc = test(net, test_loader)
    TestAcc.append(test_acc)
    print(f"Adam, epoch {epoch + 1}, best test accuracy {max(TestAcc)}")

    opt.param_groups[0]["lr"] -= lr0 / num_epochs

total_time = time.time() - t0

ax1.plot(
    torch.arange(1, len(TrainLoss) + 1).cpu() * total_time / len(TrainLoss),
    TrainLoss,
)
ax2.plot(
    torch.arange(1, len(TestAcc) + 1).cpu() * total_time / len(TestAcc),
    TestAcc,
)


"""
PSGD 
"""
net = copy.deepcopy(Net).to(device)

lr0 = 1e-3  # keep the same as Adam
opt = psgd.Affine(
    net.parameters(),
    preconditioner_init_scale=1.0,
    preconditioner_max_skew=10.0,  # use diag preconditioner for the larger dim
    lr_params=lr0,  # will aneal to lr0/num_epochs
    lr_preconditioner=0.1,  # will anneal to 0.01
    preconditioner_update_probability=1.0,  # will anneal to 0.01
    momentum=0.9,
    preconditioner_type="whitening",
)

TrainLoss = []
TestAcc = []

t0 = time.time()
for epoch in range(num_epochs):
    """train"""
    net.train()
    for _, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        def closure():
            outputs = net(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            return loss

        loss = opt.step(closure)
        TrainLoss.append(loss.item())

    """test"""
    net.eval()
    test_acc = test(net, test_loader)
    TestAcc.append(test_acc)
    print(f"PSGD, epoch {epoch + 1}, best test accuracy {max(TestAcc)}")

    opt.lr_params -= lr0 / num_epochs
    opt.lr_preconditioner = max(0.01, 0.1**0.05 * opt.lr_preconditioner)
    opt.preconditioner_update_probability = max(
        0.01, 0.1**0.1 * opt.preconditioner_update_probability
    )

total_time = time.time() - t0

ax1.plot(
    torch.arange(1, len(TrainLoss) + 1).cpu() * total_time / len(TrainLoss),
    TrainLoss,
)
ax2.plot(
    torch.arange(1, len(TestAcc) + 1).cpu() * total_time / len(TestAcc),
    TestAcc,
)


ax1.set_xlabel("Wall time (s)", fontsize=6)
ax1.set_ylabel("Train loss", fontsize=6)
ax1.tick_params(labelsize=6)
ax1.legend(
    [
        "Adam",
        "PSGD",
    ],
    fontsize=7,
)
ax1.set_title("(a)", fontsize=7)

ax2.set_xlabel("Wall time (s)", fontsize=6)
ax2.set_ylabel("Test accuracy", fontsize=6)
ax2.tick_params(labelsize=6)
ax2.legend(
    [
        "Adam",
        "PSGD",
    ],
    fontsize=7,
)
ax2.set_ylim([min(TestAcc), max(TestAcc)])
ax2.set_title("(b)", fontsize=7)

plt.savefig("vit_adam_vs_psgd.svg")
plt.show()
