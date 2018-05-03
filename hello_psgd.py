"""A 'hello world' example of PSGD on minimizing Rosenbrock function
"""
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd

def Rosenbrock(x):
    return 100.0*(x[1] - x[0]**2)**2 + (1.0 - x[0])**2

x = [torch.tensor(-1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)]
Q = 0.1*torch.eye(2)   # initialize Q with small values; otherwise, diverge
Cost = []
for i in range(300):
    cost = Rosenbrock(x)
    Cost.append(cost.item())
    g = grad(cost, x, create_graph=True)
    v = [torch.randn([]), torch.randn([])]
    gv = g[0]*v[0] + g[1]*v[1]
    hv = grad(gv, x)
    with torch.no_grad():
        Q = psgd.update_precond_dense(Q, v, hv, 0.2)
        pre_g = psgd.precond_grad_dense(Q, g)
        x[0] -= 0.5*pre_g[0]
        x[1] -= 0.5*pre_g[1]
    
plt.semilogy(Cost)