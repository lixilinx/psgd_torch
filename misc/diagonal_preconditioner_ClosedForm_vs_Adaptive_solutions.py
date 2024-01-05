"""
The diagonal preconditioner is re-discovered in many places.
P for this simplest form of PSGD has closed-form solution:

    Newton type:                P = 1/sqrt(E[h^2]) where v ~ N(0, 1) and h = H*v
    Gradient whitening type:    P = 1/sqrt(E[g^2]) where g is the gradient

Still, this example shows that to reach the same steady preconditioner fitting error, fitting P on Lie group converges faster.
Nevertheless, closed-form solution does have faster initial convergence if the Hessian does not change at all.
"""

import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0")
N = 1000000 # number of params 
num_iterations = 5000


if torch.randn([]) < 1 / 3:
    H = torch.rand(N, device=device)  # diagonal hessian, eigs ~ U[0, 1]
    print("draw eigenvalues from Uniform distribution")
elif torch.randn([]) < 1 / 2:
    H = torch.empty(N, device=device).exponential_()  # diagonal hessian, eigs ~ Exp
    print("draw eigenvalues from Exponential distribution")
else:
    H = torch.exp(torch.randn(N, device=device))  # diagonal hessian, eigs ~ LogNormal
    print("draw eigenvalues from LogNormal distribution")

"""
fitting on the group of diagonal matrix
"""
init_scale = (N / torch.sum(H * H)) ** 0.25
for step in [0.05, 0.5,]:
    Q = init_scale * torch.ones(N, device=device)
    Loss = []
    for i in range(num_iterations):
        v = torch.randn(N, device=device)
        h = H * v
        grad = Q * Q * h * h - v * v / (Q * Q)
        Q = (1 - step * grad / torch.max(torch.abs(grad))) * Q
        
        P = Q * Q
        loss = torch.sum(P * H * H + 1 / P - 2 * H)
        Loss.append(loss.item())

    plt.semilogy(Loss)


"""
closed-form solution for P by replacing E[.] with moving average.
"""
for beta in [0.999, 0.99,]:
    hh = torch.zeros(N, device=device) + 2**(-23)
    Loss = []
    for i in range(num_iterations):
        v = torch.randn(N, device=device)
        h = H * v
        if beta > i / (i + 1):
            hh = i / (i + 1) * hh + 1 / (i + 1) * h * h
        else:
            hh = beta * hh + (1 - beta) * h * h

        P = hh ** (-0.5)
        loss = torch.sum(P * H * H + 1 / P - 2 * H)
        Loss.append(loss.item())

    plt.semilogy(Loss)

plt.xlabel("Iteration")
plt.ylabel("Preconditioner fitting loss")
plt.title("Diagonal preconditioner: adaptive vs closed-form solutions")
plt.legend(
    [
        "adaptive on Lie group, lr=0.05",
        "adaptive on Lie group, lr=0.5",
        "closed-form solution, beta=0.999",
        "closed-form solution, beta=0.99",
    ]
)
plt.show()
