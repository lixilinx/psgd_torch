"""
Demo the usages of all the implemented Newton-type preconditioners on the classic Tensor Rank Decomposition problem
"""
import copy
import time

import matplotlib.pyplot as plt
import psgd
import torch

torch.set_default_device(torch.device("cuda:0"))

for mc_trial in range(100):
    # let's try a bunch of MC runs.
    R, I, J, K = 10, 20, 50, 100

    xyz0 = [
        torch.randn(R, I),  # the truth for decomposition
        torch.randn(R, J),
        torch.randn(R, K),
    ]

    T = torch.einsum("ri, rj, rk->ijk", xyz0[0], xyz0[1], xyz0[2])  # the target tensor

    xyz0 = [
        torch.randn(R, I),  # now as the initial guess for the decomposition
        torch.randn(R, J),
        torch.randn(R, K),
    ]

    def f(x, y, z):  # the decomposition loss
        Reconstructed = torch.einsum("ri, rj, rk->ijk", x, y, z)
        err = T - Reconstructed
        return torch.sum(err * err)

    num_iterations = 2000
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.yaxis.tick_right()
    ax2.yaxis.tick_right()

    """
    Gradient descent as a base line (sometimes works very well; sometimes not at all. not quite reliable)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = torch.optim.SGD(
        xyz, lr=0.0001
    )  # diverges easily with lr=0.0005; doesn't make progress with lr=0.0002
    f_values = []
    t0 = time.time()
    for epoch in range(num_iterations):
        opt.zero_grad()
        f_value = f(*xyz)
        f_values.append(f_value.item())
        f_value.backward()
        opt.step()
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    LBFGS as one more base line (more reliable than SGD, but too slow per step, and fails occasionally.)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = torch.optim.LBFGS(
        xyz, lr=0.1, max_iter=10, history_size=10
    )  # diverges easily with lr=0.5; diverges occasionally with lr=0.2
    f_values = []
    t0 = time.time()
    for epoch in range(num_iterations):

        def closure():
            opt.zero_grad()
            f_value = f(*xyz)
            f_value.backward()
            return f_value

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Dense matrix preconditioner (Only for problems with roughly 100K or less params, also it needs a lot of steps to fit the Hessian.)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.DenseNewton(
        xyz, lr_params=1.0, lr_preconditioner=0.5
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz)

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Low-rank approximation preconditioner 
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.LRANewton(
        xyz, lr_params=0.5, lr_preconditioner=0.2
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz)

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Kronecker product preconditioner 
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.KronNewton(
        xyz, preconditioner_max_skew=float("inf"), lr_params=0.2, lr_preconditioner=0.1
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz)

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Fitting loss")
    ax1.tick_params(labelsize=7)
    ax1.legend(
        [
            "Gradient descent",
            "LM-BFGS",
            "PSGD-Dense",
            "PSGD-LRA",
            "PSGD-Kron",
        ],
        fontsize=8,
    )
    ax1.set_title("Tensor rank decomposition benchmark", loc="left")

    ax2.set_xlabel("Wall time (s)")
    ax2.tick_params(labelsize=7)
    # ax2.set_ylabel("Fitting loss")
    ax2.legend(
        [
            "Gradient descent",
            "LM-BFGS",
            "PSGD-Dense",
            "PSGD-LRA",
            "PSGD-Kron",
        ],
        fontsize=8,
    )

    plt.savefig(f"psgd_vs_bfgs_trial{mc_trial}.svg")
    plt.show()