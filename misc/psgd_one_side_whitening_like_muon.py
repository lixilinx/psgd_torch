import sys 
import torch
sys.path.append("..")

from psgd import norm_lower_bound_herm

def muon_style_psgd(G, GhG, Q, lr_preconditioner=0.1, preconditioner_update_probability=1.0):
    """
    Muon style PSGD whitens gradient/moment along the smallest dim. 
    Args:
        G: 2D gradient tensor;
        GhG: EMA of G.H @ G;
        Q: factor of preconditioner P = Q.H @ Q;
        lr_preconditioner: learning rate of preconditioner; 
        preconditioner_update_probability: probability for update Q. 
        
    Outputs:
        precondG: preconditioned gradient;
        GhG: updated EMA of G.H @ G;
        Q: (optionally) updated factor of preconditioner P = Q.H @ Q. 
    """
    m, n = G.shape # assert G.dim() == 2 implicitly
    transposed = False
    if m < n:
        G = G.H
        m, n = n, m
        transposed = True
        
    # init GhG and Q on the fly if they are None. 
    if GhG is None:
        GhG = G.H @ G
    else:
        GhG = 0.9 * GhG + 0.1 * G.H @ G
    if Q is None:
        Q = torch.eye(n, device=G.device, dtype=G.dtype) * (n/torch.trace(GhG))**0.25
        
    precondG = G @ Q.H @ Q
    if torch.rand([]) < preconditioner_update_probability:
        term1 = precondG @ Q.H
        term1, term2 = term1.H @ term1, Q @ Q.H
        lr = lr_preconditioner/norm_lower_bound_herm(term1 + term2)
        Q = Q - lr * (term1 - term2) @ Q 
    
    if transposed:
        return (precondG.H, GhG, Q)
    else:
        return (precondG, GhG, Q)
    

if __name__ == "__main__":
    # test with tall real matrix
    GhG = None  # EMA of G.T @ G 
    Q = None    # preconditioner Q 
    R = 0.0     # covariance matrix of whitened G
    A = torch.randn(4, 4)
    for _ in range(1000):
        G = A @ torch.randn(4, 8)
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q)
        R = 0.9*R + 0.1*whitenedG @ whitenedG.T
    print(f"Covariance matrix of whitened G (should be close to I): \n {R.numpy()}\n")
    
    # test with short complex matrix
    GhG = None  
    Q = None
    R = 0.0  
    A = torch.randn(4, 4, dtype=torch.complex64)  
    for _ in range(1000):
        G = torch.randn(8, 4, dtype=torch.complex64) @ A
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q)
        R = 0.9*R + 0.1*whitenedG.H @ whitenedG
    print(f"|Covariance matrix| of whitened G (should be close to I): \n {torch.abs(R).numpy()}")
    
    # test with ill-conditioned matrix
    import scipy 
    import matplotlib.pyplot as plt
    G = torch.from_numpy(scipy.linalg.hilbert(64))
    G.diagonal().add_(1e-7)
    GhG = None 
    Q = None
    R = 0.0
    for _ in range(32):
        whitenedG, GhG, Q = muon_style_psgd(G, GhG, Q, lr_preconditioner=0.5)
    plt.imshow(whitenedG)
    plt.title("Whitened gradient should be close to I for PSD gradient")