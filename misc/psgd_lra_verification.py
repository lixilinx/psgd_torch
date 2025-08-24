import sys
import torch
import matplotlib.pyplot as plt 

sys.path.append("..")
import psgd

torch.set_default_dtype(torch.float64)
N, r = 10, 5
num_iterations = 100000

U = torch.randn(N, r) / N**0.5
H = torch.diag(torch.rand(N)) + U @ U.T # the true Hess 

"""
test the LRA whitening preconditioner 
"""
UVd = [] # saves states [U, V, d] for representation Q = (I + UV^T) * diag(d)
UVd.append(torch.randn(N, r))
UVd[0] *= 0.1**0.5 / torch.linalg.vector_norm(UVd[0])
UVd.append(torch.randn(N, r))
UVd[1] *= 0.1**0.5 / torch.linalg.vector_norm(UVd[1])
UVd.append(torch.ones(N, 1))

Luvd = [torch.zeros([]) for _ in range(3)] # saves L-smoothness states for [U, V, d]

errs = []
for i in range(num_iterations):
    v = torch.randn(N, 1)
    g = H @ v
    psgd.update_precond_lra_whiten(UVd, Luvd, g, lr=0.01, damping=0.0)
    precond_grad = psgd.precond_grad_lra(UVd, g)
    err = torch.linalg.vector_norm(precond_grad - v)
    errs.append(err.item())
plt.semilogy(errs)
plt.title("LRA Whitening")
plt.ylabel(r"$\|Pg - v\|$")
plt.show() 


"""
test the LRA Newton preconditioner 
"""
UVd = [] # saves states [U, V, d] for representation Q = (I + UV^T) * diag(d)
UVd.append(torch.randn(N, r))
UVd[0] *= 0.1**0.5 / torch.linalg.vector_norm(UVd[0])
UVd.append(torch.randn(N, r))
UVd[1] *= 0.1**0.5 / torch.linalg.vector_norm(UVd[1])
UVd.append(torch.ones(N, 1))

Luvd = [torch.zeros([]) for _ in range(3)] # saves L-smoothness states for [U, V, d]

errs = []
for i in range(num_iterations):
    v = torch.randn(N, 1)
    g = H @ v
    psgd.update_precond_lra_newton(UVd, Luvd, v, g, lr=0.1, damping=0.0)
    precond_grad = psgd.precond_grad_lra(UVd, g)
    err = torch.linalg.vector_norm(precond_grad - v)
    errs.append(err.item())
plt.semilogy(errs)
plt.title("LRA Newton")
plt.ylabel(r"$\|Pg - v\|$")
plt.show() 