"""
Test the PSGD Kronecker product preconditioner. A total of eight types of Q are tested:
    1, scalar
    2, diag matrix
    3, tri matrix
    4, kron(diag, diag)
    5, kron(diag, tri)
    6, kron(tri, diag)
    7, kron(tri, tri)
    8, kron(tri, tri, tri)
"""

import sys
import opt_einsum
import torch

sys.path.append("..")
from preconditioned_stochastic_gradient_descent import init_Q_exprs, update_precond_kron_math_, precond_grad_kron_math

num_iterations = 10000

#%% 
print("Test case: scalar preconditioner ")  
H = torch.complex(torch.rand([]), torch.zeros([]))
V = torch.randn([], dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, 0.0, 0.0)

for i in range(num_iterations):
    V = torch.randn([], dtype=torch.complex64)
    G = H * V 
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V; there is a bias
    # update_precond_kron_math_(Q, exprs, torch.randn([], dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening 
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: diag preconditioner ")
H = torch.complex(torch.rand(10), torch.zeros(10))
V = torch.randn(10, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, 0.0, 0.0)

for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H * V 
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(10, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%%
print("Test case: tri preconditioner ")
H = torch.randn(10, 10, dtype=torch.complex64)
H = H @ H.H
V = torch.randn(10, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, float("inf"), float("inf"))

for i in range(num_iterations):
    V = torch.randn(10, dtype=torch.complex64)
    G = H @ V 
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(10, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: kron(diag, diag) preconditioner ") 
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.complex(torch.rand(1, 3), torch.zeros(1, 3))
V = torch.randn(10, 3, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, 0.0, 0.0)
for i in range(num_iterations):
    V = torch.randn(10, 3, dtype=torch.complex64)
    G = H1 * V * H2
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton 
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(10, 3, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")
    
    
#%% 
print("Test case: kron(diag, tri) preconditioner")
H1 = torch.complex(torch.rand(10, 1), torch.zeros(10, 1))
H2 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H2 = H2 @ H2.H
V = torch.randn(10, 5, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, 7.0, float("inf"))

for i in range(num_iterations):
    V = torch.randn(10, 5, dtype=torch.complex64)
    G = H1 * V @ H2
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton 
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(10, 5, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%%
print("Test case: kron(tri, diag) preconditioner  ")
H1 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H1 = H1 @ H1.H
H2 = torch.complex(torch.rand(1, 10), torch.zeros(1, 10))
V = torch.randn(5, 10, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, 7.0, float("inf"))
for i in range(num_iterations):
    V = torch.randn(5, 10, dtype=torch.complex64)
    G = H1 @ V * H2
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton 
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(5, 10, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%% 
print("Test case: kron(tri, tri) preconditioner ") 
H1 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H2 = torch.randn(7, 7, dtype=torch.complex64) / 7**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
V = torch.randn(5, 7, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, float("inf"), float("inf"))

for i in range(num_iterations):
    V = torch.randn(5, 7, dtype=torch.complex64)
    G = H1 @ V @ H2
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton 
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(5, 7, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)} \n")


#%%  
print("Test case: kron(tri, tri, tri) preconditioner ")
H1 = torch.randn(3, 3, dtype=torch.complex64) / 3**0.5
H2 = torch.randn(4, 4, dtype=torch.complex64) / 4**0.5
H3 = torch.randn(5, 5, dtype=torch.complex64) / 5**0.5
H1 = H1 @ H1.H
H2 = H2 @ H2.H
H3 = H3 @ H3.H

V = torch.randn(3,4,5, dtype=torch.complex64)
Q, exprs = init_Q_exprs(V, 1.0, float("inf"), float("inf"))

for i in range(num_iterations):
    V = torch.randn(3,4,5, dtype=torch.complex64)
    G = opt_einsum.contract("li,mj,nk,ijk->lmn", H1,H2,H3, V)
    
    update_precond_kron_math_(Q, exprs, V, G, 0.5, "2nd", 0.0) # Newton 
    # update_precond_kron_math_(Q, exprs, None, G, (1-i/num_iterations), "2nd", 0.0) # whitening; drop V
    # update_precond_kron_math_(Q, exprs, torch.randn(3,4,5, dtype=torch.complex64), G, (1-i/num_iterations), "2nd", 0.0) # whitening
    
precond_grad = precond_grad_kron_math(Q, exprs, G)
print(f"|Preconditioned gradient - inv(H)*gradient| should be small: \n {torch.abs(precond_grad - V)}")
