"""
This example studies preconditioner fitting with finite precision arithemetic. 
Sample results:
    https://drive.google.com/file/d/1PXnFIVP480k3ACfy3583C_RX4hwUCGUO/view?usp=drive_link

1, the case with hvp

We can use a tiny L2 regularization term to lower bound the Hessian if needed. 
Alternatively, adding small noises to hvp also damps the preconditioner estimation (shown here).

2, the case with gradient only

Also, we can add small noises to the gradient to damp the gradient whitening preconditioner estimation if needed. 
Theoretically, we need to add noise ~ N(0, eps(1)) if gradient has variance 1. 
Practically, smaller noises also work as still quite a few noise samples will exceed level sqrt(eps(1)). 
"""
import sys
import matplotlib.pyplot as plt 
import torch
import opt_einsum

sys.path.append("..")
from preconditioned_stochastic_gradient_descent import init_Q_exprs, update_precond_kron_math_, precond_grad_kron_math, damped_pair_vg

num_iterations = 10000
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
    if dtype==torch.float64:
        damps = [0.0,]
    else:
        damps = [2**(-15), 2**(-14), 2**(-13), 2**(-12), 2**(-11),]
    for damp in damps:        
        V = torch.randn(N, N, N, N, N, dtype=dtype)
        Q, exprs = init_Q_exprs(V, 1.0, float("inf"), float("inf"))
        errs = []
        for i in range(num_iterations):
            V = torch.randn(N, N, N, N, N, dtype=dtype)
            G = opt_einsum.contract("aA,bB,cC,dD,eE, ABCDE->abcde", H1,H2,H3,H4,H5, V)  
            update_precond_kron_math_(Q, exprs, V, G + damp*torch.mean(torch.abs(G))*V, 0.1, "2nd", 0.0)
            precond_grad = precond_grad_kron_math(Q, exprs, G)
            err = torch.mean((precond_grad - V)**2).item()
            errs.append(err)
            if err > 10*errs[0]:
                break
        plt.semilogy(errs)
        legends.append(str(dtype)[6:] + ", damp " + str(damp))
plt.ylabel(r"$\|Pg-H^{-1}g\|^2/N^5$")
plt.xlabel("Iterations")
plt.legend(legends)
plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
plt.ylim(top=2*errs[0])
plt.title("Preconditioner fitting with hvps")
plt.savefig("psgd_hvp_with_fpa.svg")
plt.show()


"""
Preconditioner fitting with gradients (gradient whitening) 
"""
legends = []
for dtype in [torch.float64, torch.float32]:
    H1, H2, H3, H4, H5 = H1.to(dtype), H2.to(dtype), H3.to(dtype), H4.to(dtype), H5.to(dtype)
    if dtype==torch.float64:
        damps = [0.0,]
    else:
        damps = [2**(-15), 2**(-14), 2**(-13), 2**(-12), 2**(-11),]
    for damp in damps:        
        V = torch.randn(N, N, N, N, N, dtype=dtype)
        Q, exprs = init_Q_exprs(V, 1.0, float("inf"), float("inf"))
        errs = []
        for i in range(num_iterations):
            V = torch.randn(N, N, N, N, N, dtype=dtype)
            G = opt_einsum.contract("aA,bB,cC,dD,eE, ABCDE->abcde", H1,H2,H3,H4,H5, V)  
            update_precond_kron_math_(Q, exprs, *damped_pair_vg(G, damp), 0.1, "2nd", 0.0)
            precond_grad = precond_grad_kron_math(Q, exprs, G)
            err = torch.mean((precond_grad - V)**2).item()
            errs.append(err)
            if err > 10*errs[0]:
                break
        plt.semilogy(errs)
        legends.append(str(dtype)[6:] + ", damp " + str(damp))
plt.ylabel(r"$\|Pg-H^{-1}g\|^2/N^5$")
plt.xlabel("Iterations")
plt.legend(legends)
plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
plt.ylim(top=2*errs[0])
plt.title("Preconditioner fitting with gradients")
plt.savefig("psgd_whitening_with_fpa.svg")
plt.show()