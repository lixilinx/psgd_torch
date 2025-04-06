import torch
from preconditioned_stochastic_gradient_descent import norm_lower_bound


def muon_style_psgd(G, Q, lr_preconditioner=1.0, preconditioner_update_probability=1.0):
    """
    The muon style PSGD fits the preconditioner in Lie group kron(Q2=Q, Q1=I) for a short matrix gradient G, 
    and group kron(Q2=I, Q1=Q) for a tall matrix gradient G. 
    It's a special case of the PSGD Kron whitening preconditioner. 
    I also integrate out the auxiliary variable V as keeping it saves nothing here. 
    The difference with muon is that muon does instantaneous whitening, while PSGD does ensemble whitening. 
    
    This function returns whitened gradient and updated preconditioner Q. 
    """
    m, n = G.shape # assert G.dim() == 2 implicitly
    if m < n:
        G = G.H
        
    A = G @ Q.H
    precondG = A @ Q
    if torch.rand([]) < preconditioner_update_probability:
        invQ = torch.linalg.solve_triangular(Q, torch.eye(min(m,n), device=G.device), upper=True)
        invQhinvQ = invQ.H @ invQ
        AhA = A.H @ A
        lr = lr_preconditioner/(norm_lower_bound(AhA + invQhinvQ) + 1.2e-38)
        Q = Q - lr * torch.triu(AhA - invQhinvQ) @ Q  
    
    if m < n:
        return (precondG.H, Q)
    else:
        return (precondG, Q)
    

if __name__ == "__main__":
    # test muon style psgd: tall real matrix
    Q = torch.eye(4)
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4)
    for _ in range(1000):
        G = A @ torch.randn(4, 64)
        whitenedG, Q = muon_style_psgd(G, Q)
        R = 0.99*R + 0.01*whitenedG @ whitenedG.T
    print(f"Covariance matrix of whitened G (should be close to I): \n {R.numpy()}")
    
    # test muon style psgd: short complex matrix
    Q = torch.eye(4, dtype=torch.complex64)
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4, dtype=torch.complex64)  
    for _ in range(1000):
        G = torch.randn(64, 4, dtype=torch.complex64) @ A
        whitenedG, Q = muon_style_psgd(G, Q)
        R = 0.99*R + 0.01*whitenedG.H @ whitenedG
    print(f"Covariance matrix of whitened G (should be close to I): \n {torch.abs(R).numpy()}")