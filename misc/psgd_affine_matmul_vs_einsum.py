"""
We compare PSGD Affine preconditioner implementations with matmul and einsum notations. 
The einsum notation is elegant, especially for dealing with higher order tensors. See the math here:
    https://drive.google.com/file/d/1CEEq7A3_l8EcPEDa_sYtqr5aMLVeZWL7/view

Previously, I tested with torch.einsum (ver 2.4), and found that it has issues while jax doesn't. 
So, I switched to opt_einsum, and it works now.  
"""

import sys

import time

import opt_einsum
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

torch.set_default_device('cuda:0')

M, N = 5, 6
H1 = torch.randn(M, M, dtype=torch.complex64) / M**0.5
H2 = torch.randn(N, N, dtype=torch.complex64) / N**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
num_iterations = 10000

"""
PSGD Affine with the matmul notation
"""
Ql, Qr = torch.eye(M, dtype=torch.complex64), torch.eye(N, dtype=torch.complex64)
for i in range(num_iterations):
    if i==10:
        t0 = time.time()
    V = torch.randn(M, N, dtype=torch.complex64)
    G = H1 @ V @ H2
    psgd.update_precond_affine_math_(Ql, Qr, V, G, 0.1, "2nd", 0.0)
precond_grad = psgd.precond_grad_affine_math(Ql, Qr, G)
print(f"Implementation with matmul takes {time.time() - t0} s", end="\n")
print(f"Left one should be proportional to an identity matrix: \n {torch.abs(Ql.H@Ql@H1)}", end="\n")
print(f"Right one should be proportional to an identity matrix: \n {torch.abs(Qr.H@Qr@H2)}", end="\n")
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)}", end="\n\n")


"""
PSGD Affine with the einsum notation.
Just take 2nd order tensor as an example, while similar math applies to lower and higher order tensors too.  
"""
Q1, Q2 = torch.eye(M, dtype=torch.complex64), torch.eye(N, dtype=torch.complex64)
exprA = opt_einsum.contract_expression("li,mj,ij->lm", Q1.shape, Q2.shape, G.shape)
exprP = opt_einsum.contract_expression("la,li,mb,mj,ij->ab", Q1.shape, Q1.shape, Q2.shape, Q2.shape, G.shape)
for i in range(num_iterations):
    if i==10:
        t0 = time.time()
    V = torch.randn(M, N, dtype=torch.complex64)
    G = H1 @ V @ H2
    # A = torch.einsum("li,mj,ij->lm", Ql, Qr, G)
    A = exprA(Q1, Q2, G)
    # invQl = torch.linalg.solve_triangular(Ql, torch.eye(M), upper=True)
    # invQr = torch.linalg.solve_triangular(Qr, torch.eye(N), upper=True)
    # conjB = torch.einsum("ij,il,jm->lm", torch.conj(V), invQl, invQr)
    conjB = torch.linalg.solve_triangular(Q1.transpose(1,0), V.conj(), upper=False).permute(1,0) # li,ij->jl
    conjB = torch.linalg.solve_triangular(Q2.transpose(1,0), conjB, upper=False).permute(1,0) # mj,jl->lm

    term1 = torch.einsum("ip,jp->ij", A, torch.conj(A))
    term2 = torch.einsum("ip,jp->ij", torch.conj(conjB), conjB)
    Q1 = Q1 - 0.1 * torch.triu(term1 - term2) / psgd.norm_lower_bound(term1 + term2) @ Q1

    term1 = torch.einsum("pi,pj->ij", A, torch.conj(A))
    term2 = torch.einsum("pi,pj->ij", torch.conj(conjB), conjB)
    Q2 = Q2 - 0.1 * torch.triu(term1 - term2) / psgd.norm_lower_bound(term1 + term2) @ Q2
precond_grad = exprP(torch.conj(Q1), Q1, torch.conj(Q2), Q2, G)
print(f"Implementation with einsum takes {time.time() - t0} s", end="\n")
print(f"Left one should be proportional to an identity matrix: \n {torch.abs(Q1.H@Q1@H1)}", end="\n")
print(f"Right one should be proportional to an identity matrix: \n {torch.abs(Q2.T@Q2.conj()@H2)}", end="\n")
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)}")

