"""
This example studies preconditioner fitting with finite precision arithemetic. 
The PSGD-Kron-QEP version is numerically stable with single precision.
But, the PSGD-Kron-EQ version is less stable as it needs tri solver to update Q.  
"""
import sys
import matplotlib.pyplot as plt 
import torch
import opt_einsum

sys.path.append("..")
from psgd import (init_kron, 
                  precond_grad_kron_whiten_qep, 
                  update_precond_kron_newton_qep, 
                  precond_grad_kron)

num_iterations = 2000
N = 10
H1 = torch.randn(N, N, dtype=torch.float64) / N**0.5
H2 = torch.randn(N, N, dtype=torch.float64) / N**0.5
H3 = torch.randn(N, N, dtype=torch.float64) / N**0.5
H4 = torch.randn(N, N, dtype=torch.float64) / N**0.5
H5 = torch.randn(N, N, dtype=torch.float64) / N**0.5
print(f"Cond(H): {(torch.linalg.cond(H1)*torch.linalg.cond(H2)*torch.linalg.cond(H3)*torch.linalg.cond(H4)*torch.linalg.cond(H5))**2}")
H1 = H1 @ H1.T
H2 = H2 @ H2.T
H3 = H3 @ H3.T
H4 = H4 @ H4.T
H5 = H5 @ H5.T

"""
Preconditioner fitting with hvp 
"""
legends = []
for dtype in [torch.float64, torch.float32]:
    H1, H2, H3, H4, H5 = H1.to(dtype), H2.to(dtype), H3.to(dtype), H4.to(dtype), H5.to(dtype)
    
    V = torch.randn(N, N, N, N, N, dtype=dtype)
    QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), dQ="QEP")
    errs = []
    for i in range(num_iterations):
        V = torch.randn(N, N, N, N, N, dtype=dtype)
        G = opt_einsum.contract("aA,bB,cC,dD,eE, ABCDE->abcde", H1,H2,H3,H4,H5, V)  
        update_precond_kron_newton_qep(QL, exprs, V, G)
        precond_grad = precond_grad_kron(QL, exprs, G)
        err = torch.mean((precond_grad - V)**2).item()
        errs.append(err)
    plt.semilogy(errs)
    legends.append(str(dtype)[6:])
plt.ylabel(r"$\|Pg-H^{-1}g\|^2/N^5$")
plt.xlabel("Iterations")
plt.legend(legends)
plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
plt.title("Preconditioner fitting with hvps")
plt.savefig("psgd_hvp_with_fpa.svg")
plt.show()


"""
Preconditioner fitting with gradients (gradient whitening) 
"""
legends = []
for dtype in [torch.float64, torch.float32]:
    H1, H2, H3, H4, H5 = H1.to(dtype), H2.to(dtype), H3.to(dtype), H4.to(dtype), H5.to(dtype)
      
    V = torch.randn(N, N, N, N, N, dtype=dtype)
    QL, exprs = init_kron(V, 1.0, float("inf"), float("inf"), dQ="QEP")
    errs = []
    for i in range(num_iterations):
        V = torch.randn(N, N, N, N, N, dtype=dtype)
        G = opt_einsum.contract("aA,bB,cC,dD,eE, ABCDE->abcde", H1,H2,H3,H4,H5, V)  
        precond_grad = precond_grad_kron_whiten_qep(QL, exprs, G)
        err = torch.mean((precond_grad - V)**2).item()
        errs.append(err)
    plt.semilogy(errs)
    legends.append(str(dtype)[6:])
plt.ylabel(r"$\|Pg-H^{-1}g\|^2/N^5$")
plt.xlabel("Iterations")
plt.legend(legends)
plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
plt.title("Preconditioner fitting with gradients")
plt.savefig("psgd_whitening_with_fpa.svg")
plt.show()
