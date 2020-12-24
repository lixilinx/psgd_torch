"""Demo the usages of all implemented preconditioners on the classic sparse Tensor Decomposition problem
"""
import matplotlib.pyplot as plt
import torch
import preconditioned_stochastic_gradient_descent as psgd 

I, J, K = 10, 20, 50
T = torch.rand(I, J, K) # the target tensor
R = 5 # rank of reconstructed tensor
xyz = [torch.randn(R, I), # initial guess for the decomposition
       torch.randn(R, J),
       torch.randn(R, K)]
[w.requires_grad_(True) for w in xyz]

def f(): # the decomposition loss 
    x, y, z = xyz
    Reconstructed = torch.sum(x[:,:,None,None]*y[:,None,:,None]*z[:,None,None,:], dim=0)
    err = T - Reconstructed
    return torch.sum(err*err) + 1e-3*torch.sum(torch.abs(torch.cat(xyz, dim=1))) # the penalty term encourages sparse decomposition

#demo_case = 'general_dense_preconditioner'
#demo_case = 'general_sparse_LU_decomposition_preconditioner'
demo_case = 'Kronecker_product_preconditioner'

f_values = []
if demo_case == 'general_dense_preconditioner':
    num_para = sum([torch.numel(w) for w in xyz])
    Q = 0.1*torch.eye(num_para)
    for _ in range(100):
        loss = f()
        f_values.append(loss.item())
        grads = torch.autograd.grad(loss, xyz, create_graph=True)
        vs = [torch.randn_like(w) for w in xyz]
        Hvs = torch.autograd.grad(grads, xyz, vs) 
        with torch.no_grad():
            Q = psgd.update_precond_dense(Q, vs, Hvs, step=0.1)
            pre_grads = psgd.precond_grad_dense(Q, grads)
            [w.subtract_(0.1*g) for (w, g) in zip(xyz, pre_grads)]   

elif demo_case == 'general_sparse_LU_decomposition_preconditioner':
    num_para = sum([torch.numel(w) for w in xyz])
    r = 10 # this is order of LU decomposition preconditioner
    # lower triangular matrix is [L1, 0; L2, diag(l3)]; L12 is [L1; L2]
    L12 = 0.1*torch.cat([torch.eye(r), torch.zeros(num_para - r, r)], dim=0)
    l3 = 0.1*torch.ones(num_para - r, 1) 
    # upper triangular matrix is [U1, U2; 0, diag(u3)]; U12 is [U1, U2]
    U12 = 0.1*torch.cat([torch.eye(r), torch.zeros(r, num_para - r)], dim=1)
    u3 = 0.1*torch.ones(num_para - r, 1) 
    
    for _ in range(200):
        loss = f()
        f_values.append(loss.item())
        grads = torch.autograd.grad(loss, xyz, create_graph=True)
        vs = [torch.randn_like(w) for w in xyz]
        Hvs = torch.autograd.grad(grads, xyz, vs) 
        with torch.no_grad():
            L12, l3, U12, u3 = psgd.update_precond_splu(L12, l3, U12, u3, vs, Hvs, step=0.1)
            pre_grads = psgd.precond_grad_splu(L12, l3, U12, u3, grads)
            [w.subtract_(0.1*g) for (w, g) in zip(xyz, pre_grads)]   
    
elif demo_case == 'Kronecker_product_preconditioner':
    # # example 1
    # Qs = [[0.1*torch.eye(R), torch.stack([torch.ones(I), torch.zeros(I)], dim=0)], # (dense, normalization) format
    #       [0.1*torch.ones(1, R), torch.eye(J)], # (scaling, dense) format
    #       [0.1*torch.ones(1, R), torch.stack([torch.ones(K), torch.zeros(K)], dim=0)],] # (scaling, normalization) format
    
    # example 2
    Qs = [[0.1*torch.stack([torch.ones(R), torch.zeros(R)], dim=0), torch.eye(I)],
          [0.1*torch.eye(R), torch.ones(1, J)],
          [0.1*torch.stack([torch.ones(R), torch.zeros(R)], dim=0), torch.ones(1, K)],]
    
    # # example 3
    # Qs = [[0.1*torch.eye(w.shape[0]), torch.eye(w.shape[1])] for w in xyz]

    for _ in range(100):
        loss = f()
        f_values.append(loss.item())
        grads = torch.autograd.grad(loss, xyz, create_graph=True)
        vs = [torch.randn_like(w) for w in xyz]
        Hvs = torch.autograd.grad(grads, xyz, vs) 
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv, step=0.1) for (Qlr, v, Hv) in zip(Qs, vs, Hvs)]
            pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
            [w.subtract_(0.1*g) for (w, g) in zip(xyz, pre_grads)]   
    
plt.semilogy(f_values)
plt.xlabel('Iterations')
plt.ylabel('Decomposition losses')