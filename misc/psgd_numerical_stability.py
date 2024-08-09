import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

# torch.set_default_device("cuda:0")

plt.figure(figsize=[7, 3])
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()
ax3.yaxis.tick_right()

N = 50
H = torch.zeros(N, N)
for i in range(N):
    H[max(i - 1, 0), i] = 0.5
    H[i, max(i - 1, 0)] = 0.5
    H[i, i] = 1
# H = torch.randn(N, N)
# H = H@H.t()/N
I = torch.eye(N)
for count, eps in enumerate([0, 1e-2]):
    if count == 0:
        ax = ax1
        num_iterations = 150000
    else:
        ax = ax2
        num_iterations = 150000

    if eps == 0:
        H1 = H.clone()
    else:
        L, U = torch.linalg.eigh((H @ H.t() + eps**2 * I).to(torch.float64))
        H1 = (U @ torch.diag(torch.sqrt(L)) @ U.t()).to(torch.float32)

    # fitting on Lie group
    Q, invQ = torch.eye(N), torch.eye(N)
    if eps == 0:
        step = 1.0
    else:
        step = 0.1
    Loss = []
    for i in range(num_iterations):
        loss = torch.linalg.matrix_norm(Q.t() @ Q @ H1 - I)
        Loss.append(loss.item())
        
        v = torch.randn(N, 1)
        h = H @ v + eps * torch.randn(N, 1)

        psgd.update_precond_newton_math_(Q, invQ, v, h, step, "2nd", 0.0)

    ax.semilogy(range(1, 1 + num_iterations), Loss, "k")
    
    # fitting on Lie group
    Q, invQ = torch.eye(N), None
    if eps == 0:
        step = 1.0
    else:
        step = 0.1
    Loss = []
    for i in range(num_iterations):
        loss = torch.linalg.matrix_norm(Q.t() @ Q @ H1 - I)
        Loss.append(loss.item())
        
        v = torch.randn(N, 1)
        h = H @ v + eps * torch.randn(N, 1)

        psgd.update_precond_newton_math_(Q, invQ, v, h, 2 * step, "2nd", 0.0)

    ax.semilogy(range(1, 1 + num_iterations), Loss, "b")

    # closed-form solution
    Loss = []
    hh = torch.eye(N)
    for i in range(num_iterations):
        L, U = torch.linalg.eigh(hh)
        # L[L < 1e-6] = 1e-6
        P = U @ torch.diag(torch.rsqrt(L)) @ U.t()
        loss = torch.linalg.matrix_norm(P @ H1 - I)
        Loss.append(loss.item())
        
        v = torch.randn(N, 1)
        h = H @ v + eps * torch.randn(N, 1)

        if (i + 1) / (i + 2) < 0.999:
            hh = (i + 1) / (i + 2) * hh + 1 / (i + 2) * (h @ h.t())
        else:
            hh = 0.999 * hh + 0.001 * (h @ h.t())

    ax.semilogy(range(1, 1 + num_iterations), Loss, "r")

    # bfgs
    Loss = []
    P = torch.eye(N)
    for i in range(num_iterations):
        loss = torch.linalg.matrix_norm(P @ H1 - I)
        Loss.append(loss.item())
        
        v = torch.randn(N, 1)
        h = H @ v + eps * torch.randn(N, 1)
        if v.t() @ h < 0:
            h = -h  # to avoid P<0

        P = (
            P
            + (v.t() @ h + h.t() @ P @ h) * (v @ v.t()) / (h.t() @ v) ** 2
            - (P @ h @ v.t() + v @ h.t() @ P) / (v.t() @ h)
        )

    ax.semilogy(range(1, 1 + num_iterations), Loss, "m")

    ax.legend(
        [
            r"PSGD, GL$(n,\mathbb{R})$",
            r"PSGD, Triangular",
            r"$P=(E[hh^T])^{-0.5}$",
            "BFGS",
        ],
        fontsize=7,
    )
    ax.set_xlabel("Iterations", fontsize=8)
    ax.tick_params(labelsize=6)
    if count == 0:
        ax.set_ylabel(r"$||PH' - I||_F$", fontsize=8)
        ax.set_title(r"(a) Clean $Hv$", fontsize=8)
    else:
        # ax.set_ylabel(r"$||PH' - I||_F$", fontsize=8)
        ax.set_title(r"(b) Noisy $Hv$", fontsize=8)


####################################################
ax = ax3
num_iterations = 100000

# fitting on Lie group, fixed step size 1
Q, invQ = torch.eye(N), torch.eye(N)
H = torch.ones(N, N) / 4
Loss = []
for i in range(num_iterations):
    loss = torch.linalg.matrix_norm(Q.t() @ Q @ H - I)
    Loss.append(loss.item())
    
    u = torch.rand(N, 1)
    H = H + u @ u.t()
    v = torch.randn(N, 1)
    h = H @ v

    psgd.update_precond_newton_math_(Q, invQ, v, h, 1.0, "2nd", 0.0)

ax.loglog(range(1, 1 + num_iterations), Loss, "k")

# fitting on Lie group, fixed step size 1
Q, invQ = torch.eye(N), None
H = torch.ones(N, N) / 4
Loss = []
for i in range(num_iterations):
    loss = torch.linalg.matrix_norm(Q.t() @ Q @ H - I)
    Loss.append(loss.item())
    
    u = torch.rand(N, 1)
    H = H + u @ u.t()
    v = torch.randn(N, 1)
    h = H @ v

    psgd.update_precond_newton_math_(Q, invQ, v, h, 1.0, "2nd", 0.0)

ax.loglog(range(1, 1 + num_iterations), Loss, "b")

# closed-form solution
Loss = []
hh = torch.eye(N)
H = torch.ones(N, N) / 4
for i in range(num_iterations):
    L, U = torch.linalg.eigh(hh)
    # L[L < 1e-6] = 1e-6
    P = U @ torch.diag(torch.rsqrt(L)) @ U.t()
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())
    
    u = torch.rand(N, 1)
    H = H + u @ u.t()
    v = torch.randn(N, 1)
    h = H @ v

    if (i + 1) / (i + 2) < 0.999:
        hh = (i + 1) / (i + 2) * hh + 1 / (i + 2) * (h @ h.t())
    else:
        hh = 0.999 * hh + 0.001 * (h @ h.t())

ax.loglog(range(1, 1 + num_iterations), Loss, "r")

# bfgs
Loss = []
P = torch.eye(N)
H = torch.ones(N, N) / 4
for i in range(num_iterations):
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())
    
    u = torch.rand(N, 1)
    H = H + u @ u.t()
    v = torch.randn(N, 1)
    h = H @ v
    if v.t() @ h < 0:
        h = -h  # to avoid P<0

    P = (
        P
        + (v.t() @ h + h.t() @ P @ h) * (v @ v.t()) / (h.t() @ v) ** 2
        - (P @ h @ v.t() + v @ h.t() @ P) / (v.t() @ h)
    )

ax.loglog(range(1, 1 + num_iterations), Loss, "m")

ax.legend(
    [
        r"PSGD, GL$(n,\mathbb{R})$",
        r"PSGD, Triangular",
        r"$P=(E[hh^T])^{-0.5}$",
        "BFGS",
    ],
    fontsize=7,
)
ax.set_xlabel("Iterations", fontsize=8)
ax.tick_params(labelsize=6)
# ax.set_ylabel(r"$||PH - I||_F$", fontsize=8)
ax.set_title(r"(c) Time-varying $H$", fontsize=8)

# plt.tight_layout()
plt.savefig("psgd_numerical_stability.svg")
plt.show()
