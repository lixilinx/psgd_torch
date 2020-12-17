"""A 'hello world' example of PSGD on minimizing the Rosenbrock function
"""
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd

xs = [torch.tensor(-1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)]

def Rosenbrock(xs):
    x1, x2 = xs
    return 100.0*(x2 - x1**2)**2 + (1.0 - x1)**2

Q = 0.1*torch.eye(2)   # the preconditioner is Q^T*Q
f_values = []
for i in range(500):
    y = Rosenbrock(xs)
    f_values.append(y.item())
    grads = grad(y, xs, create_graph=True) # gradient
    vs = [torch.randn([]), torch.randn([])] # a random vector
    grad_vs = grads[0]*vs[0] + grads[1]*vs[1] # gradient-vector inner product
    hess_vs = grad(grad_vs, xs) # Hessian-vector product
    with torch.no_grad():
        Q = psgd.update_precond_dense(Q, vs, hess_vs, 0.2) # update the preconditioner
        pre_grads = psgd.precond_grad_dense(Q, grads) # calculate the preconditioned gradient
        [x.subtract_(0.5*g) for (x, g) in zip(xs, pre_grads)] # update the variables 
    
plt.semilogy(f_values)
plt.xlabel('Iterations')
plt.ylabel('Function values')