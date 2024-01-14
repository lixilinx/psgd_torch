"""
Accurately bounding the spectral norm of a matrix is essential for the good performance of preconditioner estimation.
Previously, PSGD uses a very loose lower bound.
The new one (updated in Jan. 2024) is significantly tighter: most likely ||A|| < 2^0.5 * ||A||_est; rarely exceeds 1.4 times.
"""
import random
import sys

import matplotlib.pyplot as plt

import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

torch.set_default_device(torch.device("cuda:0"))

ratios = []
for _ in range(10000):
    p = random.choice([1, 10, 100, 1000,])
    q = random.choice([1, 10, 100, 1000,])
    pdf = random.choice(["normal", "uniform", "exp", "cauchy", "lognormal", "geometric", "bernoulli"])

    if pdf == "cauchy":
        A = torch.empty(p, q).cauchy_() 
    elif pdf == "uniform":
        A = torch.rand(p, q)
    elif pdf == "lognormal":
        A = torch.empty(p, q).log_normal_() 
    elif pdf == "exp":
        A = torch.empty(p, q).exponential_()
    elif pdf == "geometric":
        A = torch.empty(p, q).geometric_(torch.rand([]))
    elif pdf == "bernoulli":
        A = torch.empty(p, q).bernoulli_()
    else: # gaussian as default 
        A = torch.randn(p, q)
        
    if torch.rand([]) < 0.5:
        A = A - torch.mean(A)

    if (torch.rand([]) < 0.5):
        A = torch.linalg.pinv(A)

    true_norm = torch.linalg.matrix_norm(A, ord=2)
    if true_norm > 0: # skip ratio if true_norm=0
        ratio = true_norm / psgd.norm_lower_bound(A)
        ratios.append(ratio.item())

plt.hist(ratios)
plt.xlabel(r"${||A||} \, / \, {||A||_{\rm est.\;lower\;bound}}$")
plt.ylabel("Frequency")
