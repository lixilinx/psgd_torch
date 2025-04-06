import sys 
import torch
sys.path.append("..")
from preconditioned_stochastic_gradient_descent import norm_lower_bound


def muon_style_psgd(G, GhG, Q, lr_preconditioner=0.1, preconditioner_update_probability=0.1):
    """
    The muon style PSGD fits the preconditioner in Lie group kron(Q2=Q, Q1=I) for a short matrix gradient G, 
    and group kron(Q2=I, Q1=Q) for a tall matrix gradient G. 
    It's a special case of the PSGD Kron whitening preconditioner. 
    I further 
        1) integrate out the auxiliary variable V as keeping it saves nothing here;
        2) keep the EMA of E[G.H @ G] as GhG (assuming G is a short matrix) as it's all we need to fit Q. 
    The difference with muon is that muon does instantaneous whitening, while PSGD does ensemble whitening. 
    
    This function returns whitened gradient, updated GhG and preconditioner Q. 
    """
    m, n = G.shape # assert G.dim() == 2 implicitly
    if m < n:
        G = G.H
        
    precondG = G @ Q.H @ Q
    GhG = 0.9 * GhG + 0.1 * G.H @ G
    if torch.rand([]) < preconditioner_update_probability:
        invQ = torch.linalg.solve_triangular(Q, torch.eye(min(m,n), device=G.device), upper=True)
        invQhinvQ = invQ.H @ invQ
        AhA = Q @ GhG @ Q.H # A is G @ Q.H
        lr = lr_preconditioner/(norm_lower_bound(AhA + invQhinvQ) + 1.2e-38)
        Q = Q - lr * torch.triu(AhA - invQhinvQ) @ Q  
    
    if m < n:
        return (precondG.H, GhG, Q)
    else:
        return (precondG, GhG, Q)
    

if __name__ == "__main__":
    # test muon style psgd: tall real matrix
    GhG = torch.eye(4)
    Q = torch.eye(4)
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4)
    for _ in range(1000):
        G = A @ torch.randn(4, 8)
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q, lr_preconditioner=0.5, preconditioner_update_probability=1.0)
        R = 0.9*R + 0.1*whitenedG @ whitenedG.T
    print(f"Covariance matrix of whitened G (should be close to I): \n {R.numpy()}")
    
    # test muon style psgd: short complex matrix
    GhG = torch.eye(4)
    Q = torch.eye(4, dtype=torch.complex64)
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4, dtype=torch.complex64)  
    for _ in range(1000):
        G = torch.randn(8, 4, dtype=torch.complex64) @ A
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q, lr_preconditioner=0.5, preconditioner_update_probability=1.0)
        R = 0.9*R + 0.1*whitenedG.H @ whitenedG
    print(f"Covariance matrix of whitened G (should be close to I): \n {torch.abs(R).numpy()}")