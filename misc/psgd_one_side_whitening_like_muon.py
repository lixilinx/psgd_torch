import sys 
import torch
sys.path.append("..")
from preconditioned_stochastic_gradient_descent import norm_lower_bound


def muon_style_psgd(G, GhG, Q, lr_preconditioner=1.0, preconditioner_update_probability=1.0):
    """
    The muon style PSGD fits the preconditioner in Lie group kron(Q2=Q, Q1=I) for a tall matrix gradient G, 
    and group kron(Q2=I, Q1=Q) for a short matrix gradient G. 
    It's a special case of the PSGD Kron whitening preconditioner. 
    I further 
        1) integrate out the auxiliary variable V as keeping it saves nothing here;
        2) keep the EMA of E[G.H @ G] as GhG (assuming G is a tall matrix) is all we need to fit Q. 
    The difference with muon is that muon does instantaneous whitening, while PSGD does ensemble whitening. 
    
    This function returns whitened gradient, updated GhG and preconditioner Q. 
    """
    m, n = G.shape # assert G.dim() == 2 implicitly
    transposed = False
    if m < n:
        G = G.H
        m, n = n, m
        transposed = True
        
    # better to set initial states GhG and Q to None 
    if GhG is None:
        GhG = G.H @ G
    else:
        GhG = 0.9 * GhG + 0.1 * G.H @ G
    if Q is None:
        Q = torch.eye(n, device=G.device) * (n/torch.trace(GhG))**0.25
        
    if torch.rand([]) < preconditioner_update_probability:
        invQ = torch.linalg.solve_triangular(Q, torch.eye(n, device=G.device), upper=True)
        invQhinvQ = invQ.H @ invQ
        AhA = Q @ GhG @ Q.H # A is G @ Q.H
        lr = lr_preconditioner/norm_lower_bound(AhA + invQhinvQ)
        Q = Q - lr * torch.triu(AhA - invQhinvQ) @ Q  
    
    precondG = G @ Q.H @ Q
    if transposed:
        return (precondG.H, GhG, Q)
    else:
        return (precondG, GhG, Q)
    

if __name__ == "__main__":
    # test muon style psgd: tall real matrix
    GhG = None # EMA of G.T @ G for tall G 
    Q = None # preconditioner Q 
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4)
    for _ in range(1000):
        G = A @ torch.randn(4, 8)
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q)
        R = 0.9*R + 0.1*whitenedG @ whitenedG.T
    print(f"Covariance matrix of whitened G (should be close to I): \n {R.numpy()}")
    
    # test muon style psgd: short complex matrix
    GhG = None # EMA of G.H @ G for tall G
    Q = None
    R = 0.0 # covariance matrix of whitened G
    A = torch.randn(4, 4, dtype=torch.complex64)  
    for _ in range(1000):
        G = torch.randn(8, 4, dtype=torch.complex64) @ A
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q)
        R = 0.9*R + 0.1*whitenedG.H @ whitenedG
    print(f"|Covariance matrix| of whitened G (should be close to I): \n {torch.abs(R).numpy()}")