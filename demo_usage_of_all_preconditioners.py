"""
Demo the usages of all implemented preconditioners on the classic Tensor Rank Decomposition problem
"""
import copy
import time

import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd
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

    num_iterations = 1000
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
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
        f_value = f(*xyz) + 2 ** (-23) * sum(
            [torch.sum(torch.rand_like(p) * p * p) for p in xyz]
        )
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
        xyz, lr=0.1
    )  # diverges easily with lr=0.5; diverges occasionally with lr=0.2
    f_values = []
    t0 = time.time()
    for epoch in range(num_iterations):

        def closure():
            opt.zero_grad()
            f_value = f(*xyz) + 2 ** (-23) * sum([torch.sum(0.5 * p * p) for p in xyz])
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
    XMat (matrix-free, a very simple preconditioner, but still can solve this problem reliably)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.XMat( # set preconditioner_init_scale to None 
        xyz, preconditioner_init_scale=None, lr_params=0.2, lr_preconditioner=0.1
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz) + 2 ** (-23) * sum(
                [torch.sum(torch.rand_like(p) * p * p) for p in opt._params_with_grad]
            )

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Newton method (Only for problems with roughly 10K or less params, also it needs a lot of steps to fit the Hessian.)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.Newton(
        xyz, preconditioner_init_scale=None, lr_params=0.5, lr_preconditioner=0.2
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz) + 2 ** (-23) * sum(
                [torch.sum(torch.rand_like(p) * p * p) for p in opt._params_with_grad]
            )

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Low-rank approximation (LRA. Very reliable and cheap.)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.LRA(
        xyz, preconditioner_init_scale=None, lr_params=0.2, lr_preconditioner=0.1
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz) + 2 ** (-23) * sum(
                [torch.sum(torch.rand_like(p) * p * p) for p in opt._params_with_grad]
            )

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    """
    Affine (or Kronecker product preconditioner. Reliable and converges fast.)
    """
    xyz = copy.deepcopy(xyz0)
    [w.requires_grad_(True) for w in xyz]
    opt = psgd.Affine(
        xyz, preconditioner_init_scale=None, lr_params=0.2, lr_preconditioner=0.1
    )

    f_values = []
    t0 = time.time()
    for _ in range(num_iterations):

        def closure():
            return f(*xyz) + 2 ** (-23) * sum(
                [torch.sum(torch.rand_like(p) * p * p) for p in opt._params_with_grad]
            )

        f_values.append(opt.step(closure).item())
    total_time = time.time() - t0
    ax1.semilogy(f_values)
    ax2.loglog(
        torch.arange(1, num_iterations + 1).cpu() * total_time / num_iterations,
        f_values,
    )

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Fitting loss")
    ax1.legend(
        [
            "Gradient descent",
            "LM-BFGS",
            "PSGD-Xmat",
            "PSGD-Newton",
            "PSGD-LRA",
            "PSGD-Affine",
        ],
        fontsize=8,
    )
    ax1.set_title("Tensor rank decomposition benchmark", loc="left")

    ax2.set_xlabel("Wall time (s)")
    ax2.set_ylabel("Fitting loss")
    ax2.legend(
        [
            "Gradient descent",
            "LM-BFGS",
            "PSGD-Xmat",
            "PSGD-Newton",
            "PSGD-LRA",
            "PSGD-Affine",
        ],
        fontsize=8,
    )

    plt.savefig(f"psgd_vs_bfgs_trial{mc_trial}.svg")
    plt.show()



# """
# Lastly, there is the incomplete LU (ILU) factorization preconditioner.
# I do not wrap it as a class yet.
# Looks like LRA is a better choice than ILU.
# """
# xyz = copy.deepcopy(xyz0)
# [w.requires_grad_(True) for w in xyz]

# num_paras = sum([torch.numel(w) for w in xyz])
# r = 10  # this is order of incomplete LU factorization preconditioner
# # lower triangular matrix is [L1, 0; L2, diag(l3)]; L12 is [L1; L2]
# L12 = 0.1 * torch.cat([torch.eye(r), torch.zeros(num_paras - r, r)], dim=0)
# l3 = 0.1 * torch.ones(num_paras - r, 1)
# # upper triangular matrix is [U1, U2; 0, diag(u3)]; U12 is [U1, U2]
# U12 = 0.1 * torch.cat([torch.eye(r), torch.zeros(r, num_paras - r)], dim=1)
# u3 = 0.1 * torch.ones(num_paras - r, 1)

# f_values = []
# for _ in range(num_iterations):
#     loss = f(*xyz) + 2 ** (-23) * sum([torch.sum(torch.rand_like(p) * p * p) for p in xyz])
#     f_values.append(loss.item())
#     grads = torch.autograd.grad(loss, xyz, create_graph=True)
#     vs = [torch.randn_like(w) for w in xyz]
#     Hvs = torch.autograd.grad(grads, xyz, vs)
#     with torch.no_grad():
#         L12, l3, U12, u3 = psgd.update_precond_splu(L12, l3, U12, u3, vs, Hvs, step=0.1)
#         pre_grads = psgd.precond_grad_splu(L12, l3, U12, u3, grads)
#         [w.subtract_(0.2 * g) for (w, g) in zip(xyz, pre_grads)]
# plt.semilogy(f_values)
# plt.xlabel("Iterations")
# plt.ylabel("Fitting loss")
# plt.legend(["PSGD-ILU",])
# plt.show()
