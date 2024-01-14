"""
A 'hello world' example of PSGD on minimizing the Rosenbrock function
"""
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd
import torch

xs = [torch.tensor(-1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)]

def Rosenbrock(xs):
    x1, x2 = xs
    return 100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2
 
"""
Create the Newton's optimizer.
Tuning of preconditioner_init_scale is necessary in general.
Here, I just set it to None to let the code pick up an initial value using the first Hessian-vector product. 
"""
opt = psgd.Newton(xs, preconditioner_init_scale=None, lr_params=1.0, lr_preconditioner=0.5) 
# opt = psgd.Newton(xs, preconditioner_init_scale=0.1, lr_params=1.0, lr_preconditioner=0.5) 

f_values = []
for _ in range(200):

    def closure():
        return Rosenbrock(xs)

    f_values.append(opt.step(closure).item())

plt.semilogy(f_values)
plt.xlabel("Iterations")
plt.ylabel(r"$f(\cdot)$")
plt.title("Quadratic convergence for Rosenbrock function")
plt.savefig("quadratic_convergence.svg")
plt.show()