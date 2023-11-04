"""
This example illustrates why PSGD generalizes better than methods like Adam from the view of information theory.
Model having smaller description length (DL) on train data, here (label given image), should perform better on test data if they have the same distribution.
The total DL is,

    total_DL = weight1 * DL(train data) + weight2 * (DL(model architecture) + DL(model params))

where both weight1 and weight2 should be smaller than 1 since
    1) the real world train samples are not i.i.d.,
    2) the neural network model typically is overparameterized.
But, in practice, it not easy to determine the effective sample size of a train set,
and the degree of overparameterization of most networks.

Back to this example, DL(model architecture) is a constant, i.e., DL(LeNet5).

For large sample sizes and assuming no overparameterization,
the converged model param ~ Normal, and thus DL(model params) becomes

    log det(Hessian) = - log det(P)

i.e., sharper/flatter minimum has large/smaller Hessian, and thus requires more/less bits to encode its params to achieve a certain level of prediction accuracy.

Although we cannot determine the ratio of weight1 to weight2 here,
still, we give Adam and PSGD enough iterations to check whether their results aligh with the information theory.

We observe the same trend when replacing Adam with other methods like RMSProp and SGD.
We select Adam as the counter example as it is prone to be trapped in sharp minima, see https://arxiv.org/pdf/2010.05627.pdf.
"""
import copy
import sys

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cuda:0")

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
        self.w1 = torch.nn.Parameter(0.1 * torch.randn(6, 1, 5, 5))
        self.b1 = torch.nn.Parameter(torch.zeros(6))
        self.w2 = torch.nn.Parameter(0.1 * torch.randn(16, 6, 5, 5))
        self.b2 = torch.nn.Parameter(torch.zeros(16))
        self.w3 = torch.nn.Parameter(0.1 * torch.randn(16 * 4 * 4, 120))  # FC
        self.b3 = torch.nn.Parameter(torch.zeros(120))
        self.w4 = torch.nn.Parameter(0.1 * torch.randn(120, 84))  # FC
        self.b4 = torch.nn.Parameter(torch.zeros(84))
        self.w5 = torch.nn.Parameter(0.1 * torch.randn(84, 10))  # FC
        self.b5 = torch.nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.conv2d(x, self.w2, bias=self.b2)
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(x.view(-1, 16 * 4 * 4).mm(self.w3) + self.b3)
        x = F.relu(x.mm(self.w4) + self.b4)
        return x.mm(self.w5) + self.b5


def train_loss(data, target):
    y = lenet5(data)
    y = F.log_softmax(y, dim=1)
    return F.nll_loss(y, target)


def test_loss():
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            y = lenet5(data)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred != target)
    return num_errs.item() / len(test_loader.dataset)


rank = 10  # the order of low rank approximation for Hessian estimation
ax1 = plt.subplot(121, projection="3d")
ax2 = plt.subplot(122, projection="3d")
for mc_trial in range(10):
    for wd_trial, wd in enumerate([0, 1e-4]):
        net = LeNet5()

        if wd_trial == 0:
            line_color = "k"
            ax = ax1
        else:
            line_color = "k"
            ax = ax2

        print("Monte Carlo trial: {}; wd: {}".format(mc_trial + 1, wd))
        print("\n")

        """
        Adam
        """
        print("Adam: ")
        lenet5 = copy.deepcopy(net).to(device)

        opt_adam = torch.optim.Adam(lenet5.parameters())

        opt_dummy = psgd.UVd(  # this dummy low rank approximation PSGD optimizer is only used for Hessian estimation
            lenet5.parameters(),
            rank_of_approximation=rank,
            preconditioner_init_scale=1.0,
            lr_params=0.0,
            lr_preconditioner=0.1,
        )

        TrainLosses, best_test_loss, LogDets = [], 1.0, []
        for epoch in range(20):
            for _, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                def closure():
                    xentropy = train_loss(data, target)
                    l2 = sum(
                        [
                            torch.sum(torch.rand_like(p) * p * p)
                            for p in opt_dummy._params_with_grad
                        ]
                    )
                    return xentropy + wd * l2, xentropy

                opt_adam.zero_grad()
                total_loss, loss = closure()
                total_loss.backward()
                opt_adam.step()

                # this dummy psgd optimizer is only used for Hessian estimation
                _, _ = opt_dummy.step(closure)

                TrainLosses.append(loss.item())
                logdet = (
                    torch.sum(torch.log(opt_dummy._d))
                    + torch.linalg.slogdet(
                        torch.eye(rank, device=device)
                        + opt_dummy._V.t() @ opt_dummy._U  # det(I+U*V') = det(I + V'*U)
                    )[1]
                )
                LogDets.append(logdet.item())
            best_test_loss = min(best_test_loss, test_loss())
            opt_adam.param_groups[0]["lr"] *= (0.1) ** (1 / 19)
            opt_dummy.lr_preconditioner *= (0.01) ** (1 / 19)
            print(
                "Epoch: {}; best test classification error rate: {}".format(
                    epoch + 1, best_test_loss
                )
            )

        test_err_adam = best_test_loss
        description_len_data_adam = (
            sum(TrainLosses[-1000:]) / 1000 * 60000
        )  # MNIST has 60000 training samples
        description_len_params_adam = -sum(LogDets[-1000:]) / 1000
        print("Train data description length: ", description_len_data_adam)
        print("Model params description length: ", description_len_params_adam)
        print("\n")

        # psgd
        print("PSGD: ")

        lenet5 = copy.deepcopy(net).to(device)

        opt = psgd.UVd(
            lenet5.parameters(),
            rank_of_approximation=rank,
            preconditioner_init_scale=1.0,
            lr_params=0.1,
            lr_preconditioner=0.1,
            momentum=0.9,   # match the momentum with Adam so that only their 'preconditioners' are different  
            grad_clip_max_norm=10.0,
        )

        TrainLosses, best_test_loss, LogDets = [], 1.0, []
        for epoch in range(20):
            for _, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                def closure():
                    xentropy = train_loss(data, target)
                    l2 = sum(
                        [
                            torch.sum(torch.rand_like(p) * p * p)
                            for p in opt._params_with_grad
                        ]
                    )
                    return xentropy + wd * l2, xentropy

                _, loss = opt.step(closure)

                TrainLosses.append(loss.item())
                logdet = (
                    torch.sum(torch.log(opt._d))
                    + torch.linalg.slogdet(
                        torch.eye(rank, device=device) + opt._V.t() @ opt._U
                    )[1]
                )
                LogDets.append(logdet.item())
            best_test_loss = min(best_test_loss, test_loss())
            opt.lr_params *= (0.01) ** (1 / 19)
            opt.lr_preconditioner *= (0.01) ** (1 / 19)
            print(
                "Epoch: {}; best test classification error rate: {}".format(
                    epoch + 1, best_test_loss
                )
            )

        test_err_psgd = best_test_loss
        description_len_data_psgd = sum(TrainLosses[-1000:]) / 1000 * 60000
        description_len_params_psgd = -sum(LogDets[-1000:]) / 1000
        print("Train data description length: ", description_len_data_psgd)
        print("Model params description length: ", description_len_params_psgd)
        print("\n")

        delta_description_len_data = (
            description_len_data_psgd - description_len_data_adam
        )
        delta_description_len_params = (
            description_len_params_psgd - description_len_params_adam
        )

        ax.plot(
            (description_len_data_adam, description_len_data_psgd),
            (description_len_params_adam, description_len_params_psgd),
            (test_err_adam, test_err_psgd),
            color=line_color,
            lw=1,
        )

        ax.plot(
            (
                description_len_data_adam + 0.97 * delta_description_len_data,
                description_len_data_psgd,
            ),
            (description_len_params_psgd, description_len_params_psgd),
            (test_err_psgd, test_err_psgd),
            color=line_color,
            lw=1,
        )
        ax.plot(
            (description_len_data_psgd, description_len_data_psgd),
            (
                description_len_params_adam + 0.97 * delta_description_len_params,
                description_len_params_psgd,
            ),
            (test_err_psgd, test_err_psgd),
            color=line_color,
            lw=1,
        )
        ax.plot(
            (description_len_data_psgd, description_len_data_psgd),
            (description_len_params_psgd, description_len_params_psgd),
            (test_err_adam + 0.97 * (test_err_psgd - test_err_adam), test_err_psgd),
            color=line_color,
            lw=1,
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="z", labelsize=7)
        ax.set_xlabel("Train cross entropy", fontsize=8)
        ax.set_ylabel(r"$\log\det({\rm Hess})$", fontsize=8)
        if ax is ax2:
            ax.set_zlabel("Test error rate", fontsize=8)
        ax.set_title(r"PSGD $\leftarrow$ Adam, weight decay={}".format(wd), fontsize=9)

plt.savefig("how_psgd_generalize.svg")
