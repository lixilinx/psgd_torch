import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

# torch.set_default_device(torch.device("cpu"))
N = 10
num_iterations = 5000

H = torch.zeros(N, N)
for i in range(N):
    H[max(i - 1, 0), i] = 0.5
    H[i, max(i - 1, 0)] = 0.5
    H[i, i] = 1
H = torch.kron(H, H)
I = torch.eye(N**2)

def RtoP(R):
    # return P = R^(-0.25)
    L, U = torch.linalg.eigh(R)
    P = U @ torch.diag(L ** (-0.25)) @ U.t()
    return P

""" Shampoo 
"""
Ql, Qr = torch.eye(N), torch.eye(N)
Rl, Rr = torch.eye(N), torch.eye(N)
Loss = []
for i in range(num_iterations):
    P = torch.kron(Ql, Qr)
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())

    v = torch.randn(N, N)
    h = torch.reshape(H @ torch.flatten(v.t()), [N, N]).t()
    if (i + 1) / (i + 2) < 0.999:
        Rl = (i + 1) / (i + 2) * Rl + 1 / (i + 2) * h @ h.t()
        Rr = (i + 1) / (i + 2) * Rr + 1 / (i + 2) * h.t() @ h
    else:
        Rl = 0.999 * Rl + 0.001 * h @ h.t()
        Rr = 0.999 * Rr + 0.001 * h.t() @ h

    Ql, Qr = RtoP(Rl), RtoP(Rr)

plt.semilogy(Loss, "r")


""" CASPR is a variation of Shampoo 
"""
Ql, Qr = torch.eye(N), torch.eye(N)
Rl, Rr = torch.eye(N), torch.eye(N)
In = torch.eye(N)
Loss = []
for i in range(num_iterations):
    Ql, Qr = RtoP(Rl), RtoP(Rr)
    P = (torch.kron(Ql, In) + torch.kron(In, Qr)) / 2
    P = P @ P
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())

    v = torch.randn(N, N)
    h = torch.reshape(H @ torch.flatten(v.t()), [N, N]).t()
    if (i + 1) / (i + 2) < 0.999:
        Rl = (i + 1) / (i + 2) * Rl + 1 / (i + 2) * h @ h.t()
        Rr = (i + 1) / (i + 2) * Rr + 1 / (i + 2) * h.t() @ h
    else:
        Rl = 0.999 * Rl + 0.001 * h @ h.t()
        Rr = 0.999 * Rr + 0.001 * h.t() @ h

plt.semilogy(Loss, "m")


"""
test fitting with PSGD Affine preconditioner, pair (g, v)
"""
Ql, Qr = torch.eye(N), torch.eye(N)
Loss = []
for i in range(num_iterations):
    P = torch.kron(Qr.t() @ Qr, Ql.t() @ Ql)
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())

    v = torch.randn(N, N)
    h = torch.reshape(H @ torch.flatten(v.t()), [N, N]).t()
    v = torch.randn(N, N)

    psgd.update_precond_affine_math_(Ql, Qr, v, h, 0.1, "2nd", 0.0)

plt.semilogy(Loss, "k")


"""
test fitting with PSGD Affine preconditioner, pair (h, v)
"""
Ql, Qr = torch.eye(N), torch.eye(N)
Loss = []
for i in range(num_iterations):
    P = torch.kron(Qr.t() @ Qr, Ql.t() @ Ql)
    loss = torch.linalg.matrix_norm(P @ H - I)
    Loss.append(loss.item())

    v = torch.randn(N, N)
    h = torch.reshape(H @ torch.flatten(v.t()), [N, N]).t()

    psgd.update_precond_affine_math_(Ql, Qr, v, h, 1.0, "2nd", 0.0)
   
plt.semilogy(Loss, "b")
    

plt.xlabel("Iterations")
plt.ylabel(r"$||PH - I||_F$")
plt.legend(["Shampoo", "CASPR", "PSGD-Affine, pair $(v,g)$", "PSGD-Affine, pair $(v,h)$"])
plt.savefig("fig5.eps")
