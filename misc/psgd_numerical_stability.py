"""
Closed-form solution vs PSGD for preconditioner estimation using Hessian-vector product.

It is not a surprise that PSGD could outperform closed-form solution as PSGD estimate inv(H) directly, and may have better quality.
More importantly, PSGD is numerically stable. 
"""
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cpu")

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax2.yaxis.tick_right()

"""
The diagonal preconditioner is re-discovered in many places.
P for this simplest form of PSGD has closed-form solution:

    Newton type:                P = 1/sqrt(E[h^2]) where v ~ N(0, 1) and h = H*v
    Gradient whitening type:    P = 1/sqrt(E[g^2]) where g is the gradient
    
This example shows that it is still beneficial fit such a simple P on Lie group.  
"""
N = 100000  # number of params
num_iterations = 10000

if torch.randn([]) < 1 / 3:
    H = torch.rand(N, device=device)  # diagonal hessian, eigs ~ U[0, 1]
    print("draw eigenvalues from Uniform distribution")
elif torch.randn([]) < 1 / 2:
    H = torch.empty(N, device=device).exponential_()  # diagonal hessian, eigs ~ Exp
    print("draw eigenvalues from Exponential distribution")
else:
    H = torch.exp(torch.randn(N, device=device))  # diagonal hessian, eigs ~ LogNormal
    print("draw eigenvalues from LogNormal distribution")
print(f"Hessian cond number: {(torch.max(H)/torch.min(H)).item()}")

"""
closed-form solution for P
"""
hh = torch.zeros(N, device=device) + 1e-8
Loss = []
for i in range(num_iterations):
    v = torch.randn(N, device=device)
    h = H * v
    hh = hh + h * h

    P = (hh / (i + 1)) ** (-0.5)
    loss = torch.sum(P * H * H + 1 / P - 2 * H)
    Loss.append(loss.item())

ax1.semilogy(Loss, "r")

"""
fitting on the group of diagonal matrix with PSGD; lr annealing from 0.1 to 0.01
"""
init_scale = (N / torch.sum(H * H)) ** 0.25
Q = init_scale * torch.ones(N, device=device)
Loss = []
for i in range(num_iterations):
    v = torch.randn(N, device=device)
    h = H * v
    grad = Q * Q * h * h - v * v / (Q * Q)
    Q = (1 - 0.1 * 0.1 ** (i / (num_iterations - 1)) * grad / torch.max(torch.abs(grad))) * Q

    P = Q * Q
    loss = torch.sum(P * H * H + 1 / P - 2 * H)
    Loss.append(loss.item())

ax1.semilogy(Loss, "k")

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Preconditioner fitting loss")
ax1.set_title("Diagonal preconditioner")
ax1.legend(["Closed-form solution", r"PSGD, lr $0.1\rightarrow 0.01$"])


"""
A dense preconditioner has closed-form solution:

    Newton type:                P = inv(sqrtm(E[h h^T])) where v ~ N(0, 1) and h = H*v
    Gradient whitening type:    P = inv(sqrtm(E[g g^T])) where g is the gradient
    
The closed-form solution could suffer a lot from numerical errors, while PSGD does not. 
"""
N = 100
num_iterations = 20000

if torch.rand([]) < 0.5:
    H = torch.randn(N, N, device=device)
else:
    H = torch.rand(N, N, device=device) - 0.5

if torch.rand([]) < 0.5:
    H = torch.linalg.inv(H)
    
H = H @ H.t()
print(f"Hessian cond number: {torch.linalg.cond(H).item()}")

"""
closed-form solution
"""
Loss = []
hh = torch.eye(N, device=device) * 1e-8
for i in range(num_iterations):
    v = torch.randn(N, 1, device=device)
    h = H @ v

    hh = hh + h @ h.t()
    L, U = torch.linalg.eigh(hh / (i + 1))
    L[L < 1e-8] = 1e-8  # could have negative eigs due to numerical error
    P = U @ torch.diag(torch.rsqrt(L)) @ U.t()
    loss = torch.trace(P @ H @ H + torch.linalg.inv(P) - 2 * H)
    Loss.append(loss.item())

ax2.semilogy(Loss, "r")

"""
PSGD with lr annealing from 0.1 to 0.01
"""
init_scale = (N / torch.trace(H @ H)) ** 0.25
Q = init_scale * torch.eye(N, device=device)
Loss = []
for i in range(num_iterations):
    v = torch.randn(N, 1, device=device)
    h = H @ v
    psgd.update_precond_newton_math_(Q, v, h, 0.1 * 0.1 ** (i / (num_iterations - 1)), 0.0)

    P = Q.t() @ Q
    loss = torch.trace(P @ H @ H + torch.linalg.inv(P) - 2 * H)
    Loss.append(loss.item())

ax2.semilogy(Loss, "k")

ax2.legend(["Closed-form solution", r"PSGD, lr $0.1\rightarrow 0.01$"])
ax2.set_xlabel("Iteration")
# ax2.set_ylabel("Preconditioner fitting loss")
ax2.set_title("Dense preconditioner")

plt.savefig("psgd_numerical_stability.svg")