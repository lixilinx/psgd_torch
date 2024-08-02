"""
We compare PSGD Affine kron(diag, diag) with Adafactor (https://arxiv.org/pdf/1804.04235). 
Both preconditioners have the same sublinear memory complexity. 
But unlike PSGD, Adafactor does not always normalize gradient variace to a unit scale.  
"""

import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

num_iterations = 2000
M, N = 100, 200

if torch.rand([]) < 0.3:
    H = 0.01 / (torch.rand(M, 1) @ torch.rand(1, N))
    hess_info = "Hessian has the assumed form in Adafactor"
else:
    H = 0.1 / torch.rand(M, N)
    hess_info = "Hessian doesn't have the assumed form in Adafactor"

# Adafactor
R, C = torch.zeros(M), torch.zeros(N)
beta = 0.99
precond_grad_variance = []
for i in range(num_iterations):
    G = H * torch.randn(M, N) # simulated stochastic gradient 
    R = beta * R + (1 - beta) * torch.sum(G * G, 1)
    C = beta * C + (1 - beta) * torch.sum(G * G, 0)
    P = torch.rsqrt(R[:, None] @ C[None, :] / torch.sum(R) / (1 - beta ** (i + 1)))
    precond_grad_variance.append(torch.mean((P * G) ** 2).cpu().item())
plt.semilogy(precond_grad_variance)

# PSGD Affine, kron(diag, diag)
Ql, Qr = torch.ones(M), torch.ones(N)
precond_grad_variance = []
for i in range(num_iterations):
    G = H * torch.randn(M, N) # simulated stochastic gradient 
    psgd.update_precond_affine_dropv_math_(Ql, Qr, G, 0.1, "2nd", 0.0)
    P = ((Ql**2)[:, None]) @ ((Qr**2)[None, :])
    precond_grad_variance.append(torch.mean((P * G) ** 2).cpu().item())
plt.semilogy(precond_grad_variance)
plt.semilogy(
    torch.arange(num_iterations).cpu().numpy(), torch.ones(num_iterations).cpu().numpy()
)
plt.xlabel("Iterations")
plt.ylabel("Varaince of preconditioned gradients")
plt.legend(["Adafactor", "PSGD Affine, kron(diag, diag)", "Targets"])
plt.title(hess_info)
