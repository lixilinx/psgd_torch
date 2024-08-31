"""
We compare PSGD Affine preconditioner implementations with matmul and einsum notations. 
The einsum notation is elegant, especially for dealing with higher order tensors. But, it is not quite practical.
We can increase the problem size to find out that 
    1) Implementation with the einsum hardly scales up due to difficulties like finding the optimal contraction order 
       for least memory or computation consumptionseasily. It's too slow even for matrices with sizes of hundreds.
    2) Implementation with matmul scales up much better. It works well for matrices with sizes of thousands. 
"""

import sys

import time

import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

# torch.set_default_device('cuda')

M, N = 3, 5
H1 = torch.randn(M, M, dtype=torch.complex64) / M**0.5
H2 = torch.randn(N, N, dtype=torch.complex64) / N**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
num_iterations = 10000

"""
PSGD Affine with the matmul notation
"""
Ql, Qr = torch.eye(M, dtype=torch.complex64), torch.eye(N, dtype=torch.complex64)
t0 = time.time()
for i in range(num_iterations):
    V = torch.randn(M, N, dtype=torch.complex64)
    G = H1 @ V @ H2
    psgd.update_precond_affine_math_(Ql, Qr, V, G, 0.1, "2nd", 0.0)
precond_grad = psgd.precond_grad_affine_math(Ql, Qr, G)
print(f"Implementation with matmul takes {time.time() - t0} s")
print(f"Left one should be proportional to an identity matrix: \n {torch.abs(Ql.H@Ql@H1)}")
print(f"Right one should be proportional to an identity matrix: \n {torch.abs(Qr.H@Qr@H2)}")
print(f"Preconditioned gradient should have about unitary variance: {torch.mean(precond_grad*torch.conj(precond_grad))}", end="\n\n")


"""
PSGD Affine with the einsum notation
"""
Ql, Qr = torch.eye(M, dtype=torch.complex64), torch.eye(N, dtype=torch.complex64)
t0 = time.time()
for i in range(num_iterations):
    V = torch.randn(M, N, dtype=torch.complex64)
    G = H1 @ V @ H2
    A = torch.einsum("li,mj,ij->lm", Ql, Qr, G)
    invQl = torch.linalg.solve_triangular(Ql, torch.eye(M), upper=True)
    invQr = torch.linalg.solve_triangular(Qr, torch.eye(N), upper=True)
    conjB = torch.einsum("ij,il,jm->lm", torch.conj(V), invQl, invQr)

    term1 = torch.einsum("ip,jp->ij", A, torch.conj(A))
    term2 = torch.einsum("ip,jp->ij", torch.conj(conjB), conjB)
    Ql = Ql - 0.1 * torch.triu(term1 - term2) / psgd.norm_lower_bound(term1 + term2) @ Ql

    term1 = torch.einsum("pi,pj->ij", A, torch.conj(A))
    term2 = torch.einsum("pi,pj->ij", torch.conj(conjB), conjB)
    Qr = Qr - 0.1 * torch.triu(term1 - term2) / psgd.norm_lower_bound(term1 + term2) @ Qr
precond_grad = torch.einsum("la,li,mb,mj,ij->ab", torch.conj(Ql), Ql, torch.conj(Qr), Qr, G)
print(f"Implementation with einsum takes {time.time() - t0} s")
print(f"Left one should be proportional to an identity matrix: \n {torch.abs(Ql.H@Ql@H1)}")
print(f"Right one should be proportional to an identity matrix: \n {torch.abs(Qr.T@Qr.conj()@H2)}")
print(f"Preconditioned gradient should have about unitary variance: {torch.mean(precond_grad*torch.conj(precond_grad))}")


