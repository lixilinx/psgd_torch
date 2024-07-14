"""
For PSGD gradient whitening preconditioner, we can integrate out the auxiliary variable v as 

    E_v[v^T * inv(P) * v] = E_v[tr( inv(P) * v * v^T )] = tr(inv(P))
    
since it is a dummy variable. 
Most of the time, keeping v as an auxiliary variable simplifies the computations a lot, similar to the Hutchinson’s trick. 
However, for the affine gradient whitening preconditioner, integrating out v is preferred for a few cases.
 
Do not integrate out this v if the Hessian-vector product pair (h, v) is used for preconditioning. 
"""

import sys
import time
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

def absm(A):
    # return abs(hermitian(A)) 
    A = (A + A.t())/2
    L, U = torch.linalg.eigh(A)
    return U @ torch.diag(torch.abs(L)) @ U.t()

torch.set_default_device(torch.device("cuda"))
num_iterations = 1000

"""
Case I: diagonal, diagonal 
"""
M, N = 100, 100_000
A = torch.rand(M)
B = torch.rand(N)

# with v 
Ql, Qr = torch.ones(M), torch.ones(N)
for i in range(num_iterations):
    if i==100: # warm up 
        t0 = time.time()
        
    G = A[:,None] * torch.randn(M, N) * B
    V = torch.randn(M, N)
    psgd.update_precond_affine_math_(Ql, Qr, V, G, 0.1, "2nd", 0.0)

print(f"walltime_diagonal_diagonal_with_v: {time.time() - t0}")
print(f"P*H: {Ql**2 * A, Qr**2 * B}", end="\n\n")

# drop v 
Ql, Qr = torch.ones(M), torch.ones(N)
for i in range(num_iterations):
    if i==100:
        t0 = time.time()
        
    G = A[:,None] * torch.randn(M, N) * B
    psgd.update_precond_affine_dropv_math_(Ql, Qr, G, 0.1, "2nd", 0.0)

print(f"walltime_diagonal_diagonal_drop_v: {time.time() - t0}")
print(f"P*H: {Ql**2 * A, Qr**2 * B}", end="\n\n\n")


"""
Case II: diagonal, dense 
"""
M, N = 100_000, 100
A = torch.rand(M)
B = absm(torch.randn(N, N))

# with v 
Ql, Qr = torch.ones(M), torch.eye(N)
for i in range(num_iterations):
    if i==100:
        t0 = time.time() 
        
    G = A[:,None] * torch.randn(M, N) @ B
    V = torch.randn(M, N)
    psgd.update_precond_affine_math_(Ql, Qr, V, G, 0.1, "2nd", 0.0)

print(f"walltime_diagonal_dense_with_v: {time.time() - t0}")
print(f"P*H: {Ql**2 * A, Qr.t() @ Qr @ B}", end="\n\n")


# drop v 
Ql, Qr = torch.ones(M), torch.eye(N)
for i in range(num_iterations):
    if i==100:
        t0 = time.time() 
        
    G = A[:,None] * torch.randn(M, N) @ B
    psgd.update_precond_affine_dropv_math_(Ql, Qr, G, 0.1, "2nd", 0.0)

print(f"walltime_diagonal_dense_drop_v: {time.time() - t0}")
print(f"P*H: {Ql**2 * A, Qr.t() @ Qr @ B}", end="\n\n\n")
    

"""
Case III: dense, diagonal 
"""
M, N = 100, 100_000
A = absm(torch.randn(M, M))
B = torch.rand(N)

# with v
Ql, Qr = torch.eye(M), torch.ones(N)
for i in range(num_iterations):
    if i==100:
        t0 = time.time()
        
    G = A @ torch.randn(M, N) * B 
    V = torch.randn(M, N)
    psgd.update_precond_affine_math_(Ql, Qr, V, G, 0.1, "2nd", 0.0)

print(f"walltime_dense_diagonal_with_v: {time.time() - t0}")
print(f"P*H: {Ql.t() @ Ql @ A, Qr**2 * B}", end="\n\n")


# drop v 
Ql, Qr = torch.eye(M), torch.ones(N)
for i in range(num_iterations):
    if i==100:
        t0 = time.time() 
        
    G = A @ torch.randn(M, N) * B 
    psgd.update_precond_affine_dropv_math_(Ql, Qr, G, 0.1, "2nd", 0.0)

print(f"walltime_dense_diagonal_drop_v: {time.time() - t0}")
print(f"P*H: {Ql.t() @ Ql @ A, Qr**2 * B}", end="\n\n\n")