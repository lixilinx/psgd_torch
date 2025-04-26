"""
A 'hello world' example of PSGD on minimizing the Rosenbrock function
"""
import matplotlib.pyplot as plt
import psgd
import torch

x = torch.zeros(100, requires_grad=True)

def Rosenbrock(x):
    x1, x2 = x[0::2], x[1::2]
    return torch.sum(100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2)

# create the optimizer
opt = psgd.DenseNewton(x, lr_params=1.0, lr_preconditioner=0.5, momentum=0.9)

f_values = []
for _ in range(2000):
    f_values.append(opt.step(lambda: Rosenbrock(x)).item())

plt.semilogy(f_values)
plt.xlabel("Iterations")
plt.ylabel(r"$f(\cdot)$")
plt.title("Rosenbrock function minimization")
plt.show()
