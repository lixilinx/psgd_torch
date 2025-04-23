"""
Test the PSGD Kronecker product preconditioner. A total of 
    (eight forms of Q) x (whitening, newton) x (dQ=EQ, dQ=QEP)
32 cases are tested, where the eight forms of Q are:
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

torch.set_default_dtype(torch.float64)

sys.path.append("..")
from psgd import (init_kron, 
                  precond_grad_kron_whiten_eq, 
                  precond_grad_kron_whiten_qep, 
                  update_precond_kron_eq, 
                  update_precond_kron_newton_qep, 
                  precond_grad_kron)

num_iterations = 10000

#%% 
print("Test case: scalar preconditioner ")  
H = torch.complex(torch.rand([]), torch.zeros([]))

V = torch.randn([], dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "EQ")
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex128)
    G = H * V 
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "QEP")
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex128)
    G = H * V 
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "EQ")
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex128)
    G = H * V 
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "QEP")
for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex128)
    G = H * V 
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%% 
print("Test case: diag preconditioner ")
H = torch.complex(torch.rand(10), torch.zeros(10))

V = torch.randn(10, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "EQ")
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex128)
    G = H * V 
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "QEP")
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex128)
    G = H * V 
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "EQ")
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex128)
    G = H * V 
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0.0, 0.0, "QEP")
for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex128)
    G = H * V 
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%%
print("Test case: matrix preconditioner ")
H = torch.randn(5, 5, dtype=torch.complex128)
H = H @ H.H

V = torch.randn(5, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, dtype=torch.complex128)
    G = H @ V 
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, dtype=torch.complex128)
    G = H @ V 
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, dtype=torch.complex128)
    G = H @ V 
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, dtype=torch.complex128)
    G = H @ V 
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%% 
print("Test case: kron(diag, diag) preconditioner ") 
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.complex(torch.rand(1, 3), torch.zeros(1, 3))

V = torch.randn(10, 3, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, 0, 0, "EQ")
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex128)
    G = H1 * V * H2
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0, 0, "QEP")
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex128)
    G = H1 * V * H2
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0, 0, "EQ")
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex128)
    G = H1 * V * H2
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 0, 0, "QEP")
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex128)
    G = H1 * V * H2
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")    
    
#%% 
print("Test case: kron(diag, matrix) preconditioner")
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.randn(5, 5, dtype=torch.complex128) / 5**0.5
H2 = H2 @ H2.H

V = torch.randn(10, 5, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex128)
    G = H1 * V @ H2
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex128)
    G = H1 * V @ H2
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex128)
    G = H1 * V @ H2
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex128)
    G = H1 * V @ H2
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%%
print("Test case: kron(matrix, diag) preconditioner  ")
H1 = torch.randn(5, 5, dtype=torch.complex128) / 5**0.5
H1 = H1 @ H1.H
H2 = torch.complex(torch.rand(1, 10), torch.zeros(1, 10))

V = torch.randn(5, 10, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex128)
    G = H1 @ V * H2
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex128)
    G = H1 @ V * H2
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex128)
    G = H1 @ V * H2
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, 7.0, float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex128)
    G = H1 @ V * H2
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%% 
print("Test case: kron(matrix, matrix) preconditioner ") 
H1 = torch.randn(5, 5, dtype=torch.complex128) / 5**0.5
H2 = torch.randn(7, 7, dtype=torch.complex128) / 7**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H

V = torch.randn(5, 7, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex128)
    G = H1 @ V @ H2
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex128)
    G = H1 @ V @ H2
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex128)
    G = H1 @ V @ H2
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex128)
    G = H1 @ V @ H2
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

#%%  
print("Test case: kron(matrix, matrix, matrix) preconditioner ")
H1 = torch.randn(3, 3, dtype=torch.complex128) / 3**0.5
H2 = torch.randn(4, 4, dtype=torch.complex128) / 4**0.5
H3 = torch.randn(5, 5, dtype=torch.complex128) / 5**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
H3 = H3 @ H3.H

V = torch.randn(3,4,5, dtype=torch.complex128)
QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex128)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    precond_grad = precond_grad_kron_whiten_eq(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex128)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G, lr=(1-i/num_iterations)/2)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "EQ")
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex128)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    update_precond_kron_eq(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")

QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), "QEP")
for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex128)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    update_precond_kron_newton_qep(QL, exprs, V, G, lr=(1-i/num_iterations)/2)
precond_grad = precond_grad_kron(QL, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")