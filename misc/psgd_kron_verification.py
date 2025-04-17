"""
Test the PSGD Kronecker product preconditioner. A total of 
    (eight types of Q) x (whitening, newton)  
are tested:
    1, scalar
    2, diag matrix
    3, matrix
    4, kron(diag, diag)
    5, kron(diag, matrix)
    6, kron(matrix, diag)
    7, kron(matrix, matrix)
    8, kron(matrix, matrix, matrix)
"""

import sys
import opt_einsum
import torch

sys.path.append("..")
from psgd import init_kron, precond_grad_kron_whiten, update_precond_kron_newton, precond_grad_kron_newton

num_iterations = 10000

#%% 
print("Test case: scalar preconditioner ")  
H = torch.complex(torch.rand([]), torch.zeros([]))

V = torch.randn([], dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, 0.0, 0.0)
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex64)
    G = H * V 
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0)
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex64)
    G = H * V 
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: diag preconditioner ")
H = torch.complex(torch.rand(10), torch.zeros(10))

V = torch.randn(10, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, 0.0, 0.0)
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H * V 
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0)
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H * V 
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%%
print("Test case: matrix preconditioner ")
H = torch.randn(10, 10, dtype=torch.complex64)
H = H @ H.H

V = torch.randn(10, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H @ V 
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

V = torch.randn(10, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H @ V 
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: kron(diag, diag) preconditioner ") 
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.complex(torch.rand(1, 3), torch.zeros(1, 3))

V = torch.randn(10, 3, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, 0, 0)
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex64)
    G = H1 * V * H2
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0, 0)
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex64)
    G = H1 * V * H2
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")
    
    
#%% 
print("Test case: kron(diag, matrix) preconditioner")
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H2 = H2 @ H2.H

V = torch.randn(10, 5, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, 7.0, float("inf"))
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex64)
    G = H1 * V @ H2
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"))
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex64)
    G = H1 * V @ H2
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%%
print("Test case: kron(matrix, diag) preconditioner  ")
H1 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H1 = H1 @ H1.H
H2 = torch.complex(torch.rand(1, 10), torch.zeros(1, 10))

V = torch.randn(5, 10, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, 7.0, float("inf"))
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex64)
    G = H1 @ V * H2
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"))
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex64)
    G = H1 @ V * H2
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: kron(matrix, matrix) preconditioner ") 
H1 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H2 = torch.randn(7, 7, dtype=torch.complex64) / 7**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H

V = torch.randn(5, 7, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex64)
    G = H1 @ V @ H2
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex64)
    G = H1 @ V @ H2
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%%  
print("Test case: kron(matrix, matrix, matrix) preconditioner ")
H1 = torch.randn(3, 3, dtype=torch.complex64) / 3**0.5
H2 = torch.randn(4, 4, dtype=torch.complex64) / 4**0.5
H3 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
H3 = H3 @ H3.H

V = torch.randn(3,4,5, dtype=torch.complex64)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex64)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    precond_grad = precond_grad_kron_whiten(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"))
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex64)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    update_precond_kron_newton(QL, exprs, V, G)
precond_grad = precond_grad_kron_newton(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")