import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cpu")
N = 20
num_iterations = 2000

"""
test fitting with group GL(n,R)
"""
if torch.rand([]) < 0.5:
    H = torch.rand(N, N, device=device)
else:
    H = torch.randn(N, N, device=device)

H = H @ H.t()
if torch.rand([]) < 0.5:
    H = torch.linalg.inv(H)

init_scale = (N / torch.trace(H @ H)) ** 0.25

loss0 = torch.trace(init_scale**2 * H @ H - 2 * H) + N / init_scale**2

for step in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]:
    Q, invQ = (
        init_scale * torch.eye(N, device=device),
        torch.eye(N, device=device) / init_scale,
    )
    Loss = []
    for i in range(num_iterations):
        P = Q.t() @ Q
        loss = torch.trace(P @ H @ H + torch.linalg.inv(P) - 2 * H)
        if loss > 10 * loss0:
            break
        else:
            Loss.append(loss.item())

        v = torch.randn(N, 1, device=device)
        h = H @ v
        psgd.update_precond_newton_math_(Q, invQ, v, h, step, "2nd", 0.0)

    if loss > loss0:
        break
    else:
        plt.semilogy(Loss)
        plt.legend(
            [
                "lr=" + str(step),
            ]
        )
        plt.xlabel("Iteration")
        plt.ylabel("Fitting loss")
        plt.title("GL(n,R)")
        plt.show()


"""
test fitting with the low-rank approximation (LRA) preconditioner
"""
r = 10
assert r < N

if torch.rand([]) < 0.5:
    print("LRA for sparse Hessian")
    if torch.rand([]) < 0.5:
        U = torch.rand(N, r, device=device) / N**0.5
    else:
        U = torch.randn(N, r, device=device) / N**0.5

    if torch.rand([]) < 0.5:
        d = torch.rand(N, device=device)
    else:
        d = torch.abs(torch.randn(N, device=device))

    H = torch.diag(d) + U @ U.t()
    if torch.rand([]) < 0.5:
        H = torch.linalg.inv(H)

    init_scale = (N / torch.trace(H @ H)) ** 0.25

    loss0 = torch.trace(init_scale**2 * H @ H - 2 * H) + N / init_scale**2
else:
    print("LRA for dense Hessian")

for step in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]:
    d = init_scale * torch.ones(N, 1, device=device)
    U = torch.randn(N, r, device=device) / (N * (r + 10)) ** 0.5
    V = torch.randn(N, r, device=device) / (N * (r + 10)) ** 0.5
    Loss = []
    for i in range(num_iterations):
        Q = (torch.eye(N, device=device) + U @ V.t()) @ torch.diag(d[:, 0])
        P = Q.t() @ Q
        loss = torch.trace(P @ H @ H + torch.linalg.inv(P) - 2 * H)
        if loss > 10 * loss0:
            break
        else:
            Loss.append(loss.item())

        v = torch.randn(N, 1, device=device)
        h = H @ v
        psgd.update_precond_UVd_math_(U, V, d, v, h, step, "2nd", 0.0)

    if loss > loss0:
        break
    else:
        plt.semilogy(Loss)
        plt.legend(
            [
                "lr=" + str(step),
            ]
        )
        plt.xlabel("Iteration")
        plt.ylabel("Fitting loss")
        plt.title("LRA (low-rank approximation)")
        plt.show()


"""
test fitting with Affine preconditioner
"""
H1 = torch.rand(N, N, device=device) / N**0.5
H2 = torch.rand(N, N, device=device) / N**0.5
H = torch.kron(H2, H1)
if torch.rand([]) < 0.5:
    print("Affine for decomposable Hessian")
else:
    print("Affine for non-decomposable Hessian")
    H += 0.1 * torch.randn(N**2, N**2, device=device) / N

if torch.rand([]) < 0.5:
    H = torch.linalg.inv(H)
H = H @ H.t()

init_scale = (N**2 / torch.trace(H @ H)) ** 0.25

loss0 = torch.trace(init_scale**2 * H @ H - 2 * H) + N**2 / init_scale**2

for step in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]:
    Ql, Qr = init_scale * torch.eye(N), torch.eye(N)
    Loss = []
    for i in range(num_iterations):
        P = torch.kron(Qr.t() @ Qr, Ql.t() @ Ql)
        loss = torch.trace(P @ H @ H + torch.linalg.inv(P) - 2 * H)
        if loss > 10 * loss0:
            break
        else:
            Loss.append(loss.item())

        v = torch.randn(N, N)
        h = torch.reshape(H @ torch.flatten(v.t()), [N, N]).t()

        psgd.update_precond_affine_math_(Ql, Qr, v, h, step, "2nd", 0.0)

    if loss > loss0:
        break
    else:
        plt.semilogy(Loss)
        plt.legend(
            [
                "lr=" + str(step),
            ]
        )
        plt.xlabel("Iteration")
        plt.ylabel("Fitting loss")
        plt.title("Affine")
        plt.show()
