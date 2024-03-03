"""
A 'hello world' example of PSGD on minimizing the Rosenbrock function
"""
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd
import torch

x = torch.zeros(50, 1, requires_grad=True)


def Rosenbrock(x):
    x1, x2 = x[0::2], x[1::2]
    return torch.sum(100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2)

"""
Create the optimizer.
"""
opt = psgd.Newton(x, preconditioner_init_scale=None, lr_params=1.0, lr_preconditioner=1.0)
# opt = psgd.Affine(x, preconditioner_init_scale=None, lr_params=1.0, lr_preconditioner=0.5)
# opt = psgd.XMat(x, preconditioner_init_scale=None, lr_params=1.0, lr_preconditioner=0.2)
# opt = psgd.LRA(x, preconditioner_init_scale=None, lr_params=1.0, lr_preconditioner=0.2)

f_values = []
for _ in range(2000):

    def closure():
        return Rosenbrock(x)

    f_values.append(opt.step(closure).item())

plt.semilogy(f_values)
plt.xlabel("Iterations")
plt.ylabel(r"$f(\cdot)$")
plt.title("Rosenbrock function minimization")
plt.show()
