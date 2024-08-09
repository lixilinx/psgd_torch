"""
Created in May, 2018
Pytorch functions for preconditioned SGD
@author: XILIN LI, lixilinx@gmail.com

Updated in Dec, 2020: 
Wrapped Kronecker product preconditioner for easy use: the code will select the proper Kronecker product  
preconditioner based on the formats of input left and right preconditioners.
Added torch.jit.script decorator by default.

Updates in 2022:
Added low-rank approximation (LRA) and XMat preconditioners. 
Wrapped LRA, XMat and Newton preconditioners as classes for easy use. 

Updates in 2023:
Added gradient whitening preconditioner. 
Replaced matrix norm lower bound max(abs(A)) with sqrt(max(max_i sum_j a_ij^2, max_j sum_i a_ij^2)).
Initialize Q to ((v^T*v)/(h^T*h))^(1/4)*I if the initial scale of Q is set to None.
Wrapped affine family as a class.

Updates in 2024 Jan:
Further tightened lower bound of a matrix spectral norm (see norm_lower_bound). 

Updates in 2024 Mar:
By default, the 2nd (previously 1st) order derivative info is used to normalized the step size for preconditioner update. 
For class Newton optimizer, also providing a choice for keeping inv(Q) via matrix inverse rank-2 update.  
Update rule for a triangular Q is modified to approximately match that on GL(n, R).
Functional usage of PSGD is to be deprecated, and not updated.  

Updates in 2024 Aug:
Reversing triu01 back to triu. 
QR approximation via triu01, i.e., 
    [I + A]_R = I + triu(A) + triu(A, 1)
is fairly accurate when ||A|| < 0.25, but causes regressions for large lr_preconditioner.
Impacted classes: Affine and Newton.    
"""

import torch


def norm_lower_bound(A):
    """
    Returns a cheap lower bound for the spectral norm of A.
    Numerical results on random matrices with a wide range of distributions and sizes suggest,
        norm(A) <= sqrt(2) * norm_lower_bound(A)
    Looks to be a very tight lower bound.
    """
    max_abs = torch.max(torch.abs(A)) # used to normalize A to avoid numerically under- or over-flow
    if max_abs > 0:
        A = A/max_abs
        aa = torch.real(A * A.conj())
        value0, i = torch.max(torch.sum(aa, dim=0), 0)
        value1, j = torch.max(torch.sum(aa, dim=1), 0)
        if value0 > value1:
            x = A[:, i].conj() @ A
            # We must have norm(x) > 0 since norm(x) >= value0 > value1 >= 0
            # Also, avoid expression norm(x*A^H)/norm(x) as x*A^H could under/over flow
            return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)
        else:
            x = A @ A[j].conj()
            # normx = torch.linalg.vector_norm(x)
            # if normx > 0:
            #     # Again, avoid expression norm(A^H*x)/norm(x) as A^H*x could under/over flow
            #     return max_abs * torch.linalg.vector_norm(A.H @ (x / normx))
            # else:  # A = 0
            #     return normx
            return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))
    else: # must have A=0
        return max_abs 
    

def woodbury_identity_(invA, U, V):
    # implements the Woodbury identity,
    #
    #   inv(A + U*V) = inv(A) - inv(A)*U*inv(I + V*inv(A)*U)*V*inv(A)
    #
    # with inplace update of invA.
    #
    # Note that using the Woodbury identity multiple times could accumulate numerical erros. 
    invAU = invA @ U
    VinvAU = V @ invAU
    I = torch.eye(VinvAU.shape[0], dtype=VinvAU.dtype, device=VinvAU.device)
    invA.sub_( invAU @ torch.linalg.solve(I + VinvAU, V@invA) )
    
    
def triu01(A):
    # it is useful as for a small A, the R of QR decomposition qr(I + A) is about I + triu(A, 0) + triu(A, 1)
    return torch.triu(A, diagonal=0) + torch.triu(A, diagonal=1)
    

###############################################################################
@torch.jit.script
def update_precond_dense(Q, dxs, dgs, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, List[Tensor], List[Tensor], float, float) -> Tensor
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner with positive diagonal entries 
    dxs: list of perturbations of parameters
    dgs: list of perturbations of gradients
    step: update step size normalized to range [0, 1] 
    _tiny: an offset to avoid division by zero 
    """
    dx = torch.cat([torch.reshape(x, [-1, 1]) for x in dxs])
    dg = torch.cat([torch.reshape(g, [-1, 1]) for g in dgs])
    
    a = Q.mm(dg)
    #b = torch.triangular_solve(dx, Q, upper=True, transpose=True)[0]
    b = torch.linalg.solve_triangular(Q.t(), dx, upper=False)

    grad = torch.triu(a.mm(a.t()) - b.mm(b.t()))
    # step0 = step/(grad.abs().max() + _tiny)
    step0 = step/(norm_lower_bound(grad) + _tiny)
              
    return Q - step0*grad.mm(Q)

@torch.jit.script
def precond_grad_dense(Q, grads):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    return preconditioned gradient using dense preconditioner
    Q: Cholesky factor of preconditioner
    grads: list of gradients
    """
    grad = [torch.reshape(g, [-1, 1]) for g in grads]
    lens = [g.shape[0] for g in grad]
    grad = torch.cat(grad)
    grad = Q.t().mm(Q.mm(grad))
    
    pre_grads = []
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(torch.reshape(grad[idx : idx + lens[i]], grads[i].shape))
        idx = idx + lens[i]
        
    return pre_grads


###############################################################################
def update_precond_kron(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    """
    Update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right update rule.
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero 
    """
    m, n = Ql.shape
    p, q = Qr.shape
    if m==n: # left is dense
        if p==q: #(dense, dense) format
            return _update_precond_dense_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p==2: # (dense, normalization) format
            return _update_precond_norm_dense(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p==1: # (dense, scaling) format
            return _update_precond_dense_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return _update_precond_norm_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p==1: # (normalization, scaling) format
            return _update_precond_norm_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return _update_precond_dense_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p==2: # (scaling, normalization) format
            return _update_precond_norm_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    else:
        raise Exception('Unknown Kronecker product preconditioner')
 
       
def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right way to precondition the gradient
    Grad: (matrix) gradient
    """
    m, n = Ql.shape
    p, q = Qr.shape
    if m==n: # left is dense
        if p==q: #(dense, dense) format
            return _precond_grad_dense_dense(Ql, Qr, Grad)
        elif p==2: # (dense, normalization) format
            return _precond_grad_norm_dense(Qr, Ql, Grad.t()).t()
        elif p==1: # (dense, scaling) format
            return _precond_grad_dense_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return _precond_grad_norm_dense(Ql, Qr, Grad)
        elif p==1: # (normalization, scaling) format
            return _precond_grad_norm_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return _precond_grad_dense_scale(Qr, Ql, Grad.t()).t()
        elif p==2: # (scaling, normalization) format
            return _precond_grad_norm_scale(Qr, Ql, Grad.t()).t()
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    else:
        raise Exception('Unknown Kronecker product preconditioner')
        

###############################################################################
@torch.jit.script
def _update_precond_dense_dense(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size normalized to range [0, 1] 
    _tiny: an offset to avoid division by zero 
    """
    max_l = torch.max(torch.diag(Ql))
    max_r = torch.max(torch.diag(Qr))
    
    rho = torch.sqrt(max_l/max_r)
    Ql /= rho
    Qr *= rho
    
    #A = Ql.mm( dG.mm( Qr.t() ) )
    #Bt = torch.triangular_solve((torch.triangular_solve(dX.t(), Qr, upper=True, transpose=True))[0].t(), 
    #                 Ql, upper=True, transpose=True)[0]
    A = torch.linalg.multi_dot([Ql, dG, Qr.t()])
    Bt = torch.linalg.solve_triangular(Ql.t(), torch.linalg.solve_triangular(Qr, dX, upper=True, left=False), upper=False)
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    
    # step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    # step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    step1 = step/(norm_lower_bound(grad1) + _tiny)
    step2 = step/(norm_lower_bound(grad2) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), Qr - step2*grad2.mm(Qr)
    
@torch.jit.script
def _precond_grad_dense_dense(Ql, Qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    #return torch.chain_matmul(Ql.t(), Ql, Grad, Qr.t(), Qr)
    return torch.linalg.multi_dot([Ql.t(), Ql, Grad, Qr.t(), Qr])
    

###############################################################################
# (normalization, dense) format Kronecker product preconditioner
@torch.jit.script
def _update_precond_norm_dense(ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (normalization, dense) Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    Qr has shape (N, N)
    ql[0] is the diagonal part of Ql
    ql[1,0:-1] is the last column of Ql, excluding the last entry
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size normalized to range [0, 1] 
    _tiny: an offset to avoid division by zero  
    """
    # make sure that Ql and Qr have similar dynamic range
    max_l = torch.max(ql[0])
    max_r = torch.max(torch.diag(Qr))  
    rho = torch.sqrt(max_l/max_r)
    ql /= rho
    Qr *= rho
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t()*dG + ql[1:].t().mm( dG[-1:] ) # Ql*dG 
    A = A.mm(Qr.t())
    
    Bt = dX/ql[0:1].t()
    Bt[-1:] -= (ql[1:]/(ql[0:1]*ql[0,-1])).mm(dX)
    #Bt = torch.triangular_solve(Bt.t(), Qr, upper=True, transpose=True)[0].t()
    Bt = torch.linalg.solve_triangular(Qr, Bt, upper=True, left=False)
    
    grad1_diag = torch.sum(A*A, dim=1) - torch.sum(Bt*Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t()) 
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])  

    step1 = step/(torch.max(torch.max(torch.abs(grad1_diag)), 
                            torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    # step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    step2 = step/(norm_lower_bound(grad2) + _tiny)
    
    return torch.stack((new_ql0, new_ql1)), Qr - step2*grad2.mm(Qr)

@torch.jit.script
def _precond_grad_norm_dense(ql, Qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (normalization, dense) Kronecker product preconditioner 
    Suppose Grad has shape (M, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    Qr: shape (N, N), Cholesky factor of right preconditioner
    Grad: (matrix) gradient
    """
    preG = ql[0:1].t()*Grad + ql[1:].t().mm(Grad[-1:]) # Ql*Grad 
    #preG = torch.chain_matmul(preG, Qr.t(), Qr)
    preG = torch.linalg.multi_dot([preG, Qr.t(), Qr])
    add_last_row = ql[1:].mm(preG) # use it to modify the last row
    preG *= ql[0:1].t()
    preG[-1:] += add_last_row
    
    return preG


###############################################################################
# (normalization, scaling) Kronecker product preconditioner 
# the left one is a normalization preconditioner; the right one is a scaling preconditioner
@torch.jit.script
def _update_precond_norm_scale(ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (normalization, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero  
    """
    # make sure that Ql and Qr have similar dynamic range
    max_l = torch.max(ql[0])
    max_r = torch.max(qr) # qr always is positive
    rho = torch.sqrt(max_l/max_r)
    ql /= rho
    qr *= rho
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t()*dG + ql[1:].t().mm( dG[-1:] ) # Ql*dG 
    A *= qr # Ql*dG*Qr 
    
    Bt = dX/ql[0:1].t()
    Bt[-1:] -= (ql[1:]/(ql[0:1]*ql[0,-1])).mm(dX)
    Bt /= qr # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = torch.sum(A*A, dim=1) - torch.sum(Bt*Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t()) 
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])  

    step1 = step/(torch.max(torch.max(torch.abs(grad1_diag)), 
                            torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = torch.sum(A*A, dim=0, keepdim=True) - torch.sum(Bt*Bt, dim=0, keepdim=True)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    
    return torch.stack((new_ql0, new_ql1)), qr - step2*grad2*qr

@torch.jit.script
def _precond_grad_norm_scale(ql, qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (normalization, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    ql has shape (2, M) 
    qr has shape (1, N) 
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    Grad: (matrix) gradient
    """
    preG = ql[0:1].t()*Grad + ql[1:].t().mm(Grad[-1:]) # Ql*Grad 
    preG *= (qr*qr) # Ql*Grad*Qr^T*Qr
    add_last_row = ql[1:].mm(preG) # use it to modify the last row
    preG *= ql[0:1].t()
    preG[-1:] += add_last_row
    
    return preG


###############################################################################
@torch.jit.script
def _update_precond_dense_scale(Ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (dense, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    Ql has shape (M, M)
    qr has shape (1, N)
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero 
    """
    max_l = torch.max(torch.diag(Ql))
    max_r = torch.max(qr)
    
    rho = torch.sqrt(max_l/max_r)
    Ql /= rho
    qr *= rho
    
    A = Ql.mm( dG*qr )
    #Bt = torch.triangular_solve(dX/qr, Ql, upper=True, transpose=True)[0]
    Bt = torch.linalg.solve_triangular(Ql.t(), dX/qr, upper=False)
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.sum(A*A, dim=0, keepdim=True) - torch.sum(Bt*Bt, dim=0, keepdim=True)
    
    # step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step1 = step/(norm_lower_bound(grad1) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), qr - step2*grad2*qr
    
@torch.jit.script
def _precond_grad_dense_scale(Ql, qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (dense, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    Ql: shape (M, M), (left side) Cholesky factor of preconditioner
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    #return torch.chain_matmul(Ql.t(), Ql, Grad*(qr*qr))
    return torch.linalg.multi_dot([Ql.t(), Ql, Grad*(qr*qr)])



###############################################################################   
@torch.jit.script                     
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=0.01, _tiny=1.2e-38):
    # type: (Tensor,Tensor,Tensor,Tensor, List[Tensor],List[Tensor], float,float) -> Tuple[Tensor,Tensor,Tensor,Tensor]
    """
    update sparse LU preconditioner P = Q^T*Q, where 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: update step size normalized to range [0, 1] 
    _tiny: an offset to avoid division by zero 
    """
    # make sure that L and U have similar dynamic range
    max_l = torch.max(torch.max(torch.diag(L12)), torch.max(l3))
    max_u = torch.max(torch.max(torch.diag(U12)), torch.max(u3))
    rho = torch.sqrt(max_l/max_u)
    L12 /= rho
    l3 /= rho
    U12 *= rho
    u3 *= rho
    
    # extract the blocks
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]
    
    dx = torch.cat([torch.reshape(x, [-1, 1]) for x in dxs]) # a tall column vector
    dg = torch.cat([torch.reshape(g, [-1, 1]) for g in dgs]) # a tall column vector
    
    # U*dg
    Ug1 = U1.mm(dg[:r]) + U2.mm(dg[r:])
    Ug2 = u3*dg[r:]
    # Q*dg
    Qg1 = L1.mm(Ug1)
    Qg2 = L2.mm(Ug1) + l3*Ug2
    # inv(U^T)*dx
    #iUtx1 = torch.triangular_solve(dx[:r], U1, upper=True, transpose=True)[0]
    iUtx1 = torch.linalg.solve_triangular(U1.t(), dx[:r], upper=False)
    iUtx2 = (dx[r:] - U2.t().mm(iUtx1))/u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2/l3
    #iQtx1 = torch.triangular_solve(iUtx1 - L2.t().mm(iQtx2), L1, upper=False, transpose=True)[0]
    iQtx1 = torch.linalg.solve_triangular(L1.t(), iUtx1 - L2.t().mm(iQtx2), upper=True)
    # L^T*Q*dg
    LtQg1 = L1.t().mm(Qg1) + L2.t().mm(Qg2)
    LtQg2 = l3*Qg2
    # P*dg
    Pg1 = U1.t().mm(LtQg1)
    Pg2 = U2.t().mm(LtQg1) + u3*LtQg2
    # inv(L)*inv(Q^T)*dx
    #iLiQtx1 = torch.triangular_solve(iQtx1, L1, upper=False)[0]
    iLiQtx1 = torch.linalg.solve_triangular(L1, iQtx1, upper=False)
    iLiQtx2 = (iQtx2 - L2.mm(iLiQtx1))/l3
    # inv(P)*dx
    iPx2 = iLiQtx2/u3
    #iPx1 = torch.triangular_solve(iLiQtx1 - U2.mm(iPx2), U1, upper=True)[0]
    iPx1 = torch.linalg.solve_triangular(U1, iLiQtx1 - U2.mm(iPx2), upper=True)
    
    # update L
    grad1 = Qg1.mm(Qg1.t()) - iQtx1.mm(iQtx1.t())
    grad1 = torch.tril(grad1)
    grad2 = Qg2.mm(Qg1.t()) - iQtx2.mm(iQtx1.t())
    grad3 = Qg2*Qg2 - iQtx2*iQtx2
    # max_abs_grad = torch.max(torch.abs(grad1))
    # max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad2)))
    # max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad3)))
    # step0 = step/(max_abs_grad + _tiny)
    step0 = step/(torch.maximum(norm_lower_bound(torch.cat([grad1, grad2], 0)), torch.max(torch.abs(grad3))) + _tiny)
    newL1 = L1 - step0*grad1.mm(L1)
    newL2 = L2 - step0*grad2.mm(L1) - step0*grad3*L2
    newl3 = l3 - step0*grad3*l3

    # update U
    grad1 = Pg1.mm(dg[:r].t()) - dx[:r].mm(iPx1.t())
    grad1 = torch.triu(grad1)
    grad2 = Pg1.mm(dg[r:].t()) - dx[:r].mm(iPx2.t())
    grad3 = Pg2*dg[r:] - dx[r:]*iPx2
    # max_abs_grad = torch.max(torch.abs(grad1))
    # max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad2)))
    # max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad3)))
    # step0 = step/(max_abs_grad + _tiny)
    step0 = step/(torch.maximum(norm_lower_bound(torch.cat([grad1, grad2], 1)), torch.max(torch.abs(grad3))) + _tiny)
    newU1 = U1 - U1.mm(step0*grad1)
    newU2 = U2 - U1.mm(step0*grad2) - step0*grad3.t()*U2
    newu3 = u3 - step0*grad3*u3

    return torch.cat([newL1, newL2], dim=0), newl3, torch.cat([newU1, newU2], dim=1), newu3

@torch.jit.script
def precond_grad_splu(L12, l3, U12, u3, grads):
    # type: (Tensor,Tensor,Tensor,Tensor, List[Tensor]) -> List[Tensor]
    """
    return preconditioned gradient with sparse LU preconditioner
    where P = Q^T*Q, 
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    grads: a list of gradients to be preconditioned
    """
    grad = [torch.reshape(g, [-1, 1]) for g in grads] # a list of column vector
    lens = [g.shape[0] for g in grad] # length of each column vector
    grad = torch.cat(grad)  # a tall column vector
    
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]    
    
    # U*g
    Ug1 = U1.mm(grad[:r]) + U2.mm(grad[r:])
    Ug2 = u3*grad[r:]
    # Q*g
    Qg1 = L1.mm(Ug1)
    Qg2 = L2.mm(Ug1) + l3*Ug2
    # L^T*Q*g
    LtQg1 = L1.t().mm(Qg1) + L2.t().mm(Qg2)
    LtQg2 = l3*Qg2
    # P*g
    pre_grad = torch.cat([U1.t().mm(LtQg1),
                          U2.t().mm(LtQg1) + u3*LtQg2])
    
    pre_grads = [] # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(torch.reshape(pre_grad[idx : idx + lens[i]], grads[i].shape))
        idx = idx + lens[i]
    
    return pre_grads



##############################################################################
#
# The low-rank approximation (LRA) preconditioner is defined as
#
#   Q = (I + U*V')*diag(d)
#
# which, after reparameterization, is equivalent to form
#
#   diag(d) + U*V'
# 
# UVd as an alias of LRA due to the form of this preconditioner. 
# 

#@torch.jit.script
def IpUVtmatvec(U, V, x):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Returns (I + U*V')*x. All variables are either matrices or column vectors. 
    """
    return x + U.mm(V.t().mm(x))

# def IpUVtsolve(U, V, x):
#     """
#     Returns inv(I + U*V')*x. All variables are either matrices or column vectors.
#     """
#     VtU = V.t().mm(U)
#     I = torch.eye(VtU.size(dim=0), dtype=VtU.dtype, device=VtU.device)
#     return x - U.mm(torch.linalg.solve(I + VtU, V.t().mm(x))) # torch.solve is slow

# def norm_UVt(U, V):
#     """
#     Returns ||U*V'||_fro = sqrt(tr(U'*U*V'*V)) = sqrt(sum((U'*U)*(V'*V))) 
#     """
#     return torch.sqrt(torch.abs(torch.sum( (U.t().mm(U))*(V.t().mm(V)) )))

#@torch.jit.script
def update_precond_UVd_math_(U, V, d, v, h, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Update preconditioner Q = (I + U*V')*diag(d) with (vector, Hessian-vector product) = (v, h).
    State variables U, V and d are updated inplace. 
                               
    U, V, d, v, and h are either matrices or column vectors.  
    """
    # balance the numerical dynamic ranges of U and V; optional 
    if torch.rand([]) < 0.01:
        normU = torch.linalg.vector_norm(U)
        normV = torch.linalg.vector_norm(V)
        rho = torch.sqrt(normU/normV)
        U.div_(rho)
        V.mul_(rho)

    Qh = IpUVtmatvec(U, V, d*h)
    Ph = d*IpUVtmatvec(V, U, Qh)
    
    # invQtv = IpUVtsolve(V, U, v/d)
    # invPv = IpUVtsolve(U, V, invQtv)/d
    VtU = V.t().mm(U)
    I = torch.eye(VtU.size(dim=0), dtype=VtU.dtype, device=VtU.device)
    IpVtU = I + VtU
    invQtv = v/d
    # torch's linalg.solve is slow for small matrix
    # invQtv = invQtv - V.mm(torch.linalg.solve(IpVtU.t(), U.t().mm(invQtv)))  
    # invPv  = invQtv - U.mm(torch.linalg.solve(IpVtU,     V.t().mm(invQtv)))
    LU, pivots = torch.linalg.lu_factor(IpVtU)
    invQtv = invQtv - V.mm(torch.linalg.lu_solve(LU, pivots, U.t().mm(invQtv), adjoint=True))
    invPv  = invQtv - U.mm(torch.linalg.lu_solve(LU, pivots, V.t().mm(invQtv)))
    invPv = invPv/d

    nablaD = Ph*h - v*invPv
    if step_normalizer == '2nd':
        mu = step*torch.min(torch.rsqrt(Ph*Ph + v*v + tiny)*torch.rsqrt(h*h + invPv*invPv + tiny)) # two seperate rsqrt's to avoid underflow
    else:
        mu = step/(torch.max(torch.abs(nablaD)) + tiny)
    # d = d - mu*d*nablaD
    d.sub_(mu*d*nablaD)
    
    # update either U or V, not both at the same time
    a, b = Qh, invQtv
    if torch.rand([]) < 0.5:
        # nablaU = Qh.mm(Qh.t().mm(V)) - invQtv.mm(invQtv.t().mm(V))
        # mu = step/(norm_UVt(nablaU, V) + _tiny)
        # U = U - mu*(nablaU + nablaU.mm(V.t().mm(U)))
        atV = a.t().mm(V)
        btV = b.t().mm(V)
        atVVt = atV.mm(V.t())
        btVVt = btV.mm(V.t())
        if step_normalizer == '2nd':
            mu = step/( torch.linalg.vector_norm(a)*torch.linalg.vector_norm(atVVt) 
                       +torch.linalg.vector_norm(b)*torch.linalg.vector_norm(btVVt) + tiny)
        else: # '1st'
            norm = torch.sqrt(torch.abs( (a.t().mm(a))*(atVVt.mm(atVVt.t())) # abs to avoid sqrt(-0.0) 
                                        +(b.t().mm(b))*(btVVt.mm(btVVt.t())) 
                                      -2*(a.t().mm(b))*(atVVt.mm(btVVt.t())) ))
            mu = step/(norm + tiny)
        # U = U - mu*( a.mm(atV.mm(IpVtU)) 
        #             -b.mm(btV.mm(IpVtU)) )
        U.sub_(mu*( a.mm(atV.mm(IpVtU)) 
                   -b.mm(btV.mm(IpVtU)) ))
    else:
        # nablaV = Qh.mm(Qh.t().mm(U)) - invQtv.mm(invQtv.t().mm(U))
        # mu = step/(norm_UVt(U, nablaV) + _tiny)
        # V = V - mu*(nablaV + V.mm(U.t().mm(nablaV)))
        atU = a.t().mm(U)
        btU = b.t().mm(U)
        UUta = U.mm(atU.t())
        UUtb = U.mm(btU.t())
        if step_normalizer == '2nd':
            mu = step/( torch.linalg.vector_norm(a)*torch.linalg.vector_norm(UUta)
                       +torch.linalg.vector_norm(b)*torch.linalg.vector_norm(UUtb) + tiny)
        else: # '1st'
            norm = torch.sqrt(torch.abs( (UUta.t().mm(UUta))*(a.t().mm(a)) # abs to avoid sqrt(-0.0)
                                        +(UUtb.t().mm(UUtb))*(b.t().mm(b))
                                      -2*(UUta.t().mm(UUtb))*(a.t().mm(b)) ))
            mu = step/(norm + tiny)
        # V = V - mu*( (a + V.mm(atU.t())).mm(atU) 
        #             -(b + V.mm(btU.t())).mm(btU) )
        V.sub_(mu*( (a + V.mm(atU.t())).mm(atU) 
                   -(b + V.mm(btU.t())).mm(btU) ))

    # return [U, V, d]

#@torch.jit.script
def precond_grad_UVd_math(U, V, d, g):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    """
    Preconditioning gradient g with Q = (I + U*V')*diag(d).
                                         
    All variables here are either matrices or column vectors. 
    """
    g = IpUVtmatvec(U, V, d*g)
    g = d*IpUVtmatvec(V, U, g)
    return g


class LRA:
    """
    Implements the low-rank approximation (LRA, UVd as an alias) preconditioner, Q = (I + U*V')*diag(d), as a class.

    Args for initialization:
        params_with_grad: a list of parameters or variables requiring gradients;
        rank_of_approximation: rank of approximation, i.e., max rank of U or V, with 0 for diagonal preconditioner;
        preconditioner_init_scale: initial scale of Q, or roughly, Q = preconditioner_init_scale*eye(), with None for automatical setting;
        lr_params: normalized learning rate for parameters in range [0, 1];
        lr_preconditioner: normalized learning rate for preconditioner in range [0, 1];
        momentum: momentum factor in range [0,1);
        grad_clip_max_norm: maximum allowable gradient norm after clipping, None for no clipping;
        preconditioner_update_probability: probability on updating Q, 1 for updating at every step, and 0 for never, i.e., SGD when Q=I;
        step_normalizer: '1st' for normalizing lr_preconditioner with 1st order derivative info, and '2nd' for normalizing with 2nd derivative info; 
        exact_hessian_vector_product: True for exact Hessian-vector product via 2nd derivative,
                                    and False for approximate one via the finite difference method;
        preconditioner_type: "Newton" or "whitening", see https://arxiv.org/abs/1809.10232 for the Newton and (empirical) Fisher types.  

    Notes:
        Note 1: The Hessian-vector product can be approximated using the finite difference method by setting 
        exact_hessian_vector_product = False when the 2nd derivatives is not available.
        In this case, make sure that the closure produces the same outputs given the same inputs, 
        except for numerical errors due to non-deterministic behaviors.
        Random numbers, if any, used inside the closure should be generated starting from the same state, where the rng state can be
        read and set by, e.g., `torch.cuda.get_rng_state' and `torch.cuda.set_rng_state', respectively.
        
        Note 2: Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate.
        This is necessary as the learning rate in PSGD is normalized. 

        Note 3: `torch.linalg.solve' is called twice in function `update_precond_UVd_math_'.
        Certain solver could be orders of magnitude faster than others, especially for small matrices 
        (see https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view, Table 2).
        Considering replace it with faster ones if the default solver is too slow.

        Note 4: Currently, no support of sparse and mixed-precision gradients. 
        Half precision (bfloat16) works well except that torch.linalg.solve (v2.2) requires casting bfloat16 to float32.    
        
        Note 5: lr_params, lr_preconditioner, momentum, grad_clip_max_norm, preconditioner_update_probability, step_normalizer, 
        and exact_hessian_vector_product (bool) all can be reset on the fly. 
    """
    def __init__(self,  params_with_grad, rank_of_approximation:int=10, preconditioner_init_scale=None,
                        lr_params=0.01, lr_preconditioner=None, momentum=0.0,
                        grad_clip_max_norm=None, preconditioner_update_probability=1.0,
                        step_normalizer='2nd',
                        exact_hessian_vector_product:bool=True, preconditioner_type="Newton"):
        # mutable members
        self.lr_params = lr_params
        if lr_preconditioner is None:
            if step_normalizer == '2nd':
                self.lr_preconditioner = 0.1
            else:
                self.lr_preconditioner = 0.01
        else:
            self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        self.exact_hessian_vector_product = exact_hessian_vector_product
        self.step_normalizer = step_normalizer
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps**0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        # check rank_of_approximation
        if rank_of_approximation <= 0:
            print("Hint: the Xmat preconditioner may be more efficinet in this case.")
        if 2*rank_of_approximation + 1 >= num_params:
            print("Hint: the Newton preconditioner may be more efficient in this case.")
        # +10 to 1) avoid /0; 2) make sure that norm(U*V') << 1 even when rank_of_approximation=1
        self._U = torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device) / (num_params*(rank_of_approximation+10))**0.5
        self._V = torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device) / (num_params*(rank_of_approximation+10))**0.5
        if preconditioner_init_scale is None:
            self._d = None # set it on the fly 
        else:
            self._d = torch.ones(num_params, 1, dtype=dtype, device=device) * preconditioner_init_scale
        self._m = None # momentum buffer 
        self._preconditioner_type = preconditioner_type


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the low-rank approximation (LRA, or UVd) preconditioner, i.e., 
        updating the trainable parameters once, and returning what closure returns.

        Args:
            closure (callable): a (stateless) closure that evaluates the function of self._params_with_grad,
                                and returns the loss, or an iterable with the first one being loss.
                                Random numbers, if any, used inside the closure should be generated starting 
                                from the same rng state if exact_hessian_vector_product=False and preconditioner_type="Newton". 
        """
        if (self._preconditioner_type=="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._d is None)):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self.exact_hessian_vector_product:
                # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(param) for param in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs)
            else:
                # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
            # update preconditioner
            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) # column vector
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) # column vector  
            # set self._d if it is None 
            if self._d is None:
                self._d = (torch.sum(v*v)/torch.sum(h*h))**0.25 * torch.ones_like(v)
            # update self._U, _V and _d
            update_precond_UVd_math_(self._U, self._V, self._d, v, h, self.lr_preconditioner, self.step_normalizer, self._tiny)
            # if self.exact_hessian_vector_product:
            #     update_precond_UVd_math_(self._U, self._V, self._d,
            #                              v[:,None], h[:,None], step=self.lr_preconditioner, tiny=self._tiny)
            # else: # compensate the levels of v and h; helpful to reduce numerical errors in half-precision training
            #     update_precond_UVd_math_(self._U, self._V, self._d,
            #                              v[:,None]/self._delta_param_scale, h[:,None]/self._delta_param_scale, step=self.lr_preconditioner, tiny=self._tiny)
        else:
            # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            vs = None # no vs and Hvs
            
        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) # column vector 
        
        # update preconditioner here if it is the whitening type 
        if (self._preconditioner_type!="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._d is None)):
            if self._d is None:
                self._d = (len(grad)/torch.sum(grad*grad))**0.25 * torch.ones_like(grad)
            # update the preconditioner whitening the gradients  
            v = torch.randn_like(grad)
            update_precond_UVd_math_(self._U, self._V, self._d, v, grad, self.lr_preconditioner, self.step_normalizer, self._tiny)

        # preconditioned gradients; momentum is optional      
        if self.momentum > 0:
            if self._m is None:
                self._m = (1 - self.momentum)*grad
            else:
                self._m.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
            pre_grad = precond_grad_UVd_math(self._U, self._V, self._d, self._m)
        else:
            self._m = None # clean the buffer when momentum is set to zero 
            pre_grad = precond_grad_UVd_math(self._U, self._V, self._d, grad)
            
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # update the parameters
        if self.exact_hessian_vector_product or (vs is None) or (self._preconditioner_type!="Newton"):
            delta = lr * pre_grad
        else: # in this case, do not forget to remove the perturbation on parameters
            delta = lr * pre_grad + v
        # -delta 
        [param.subtract_(delta[j - i:j].view_as(param))
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns

# UVd as an alias
UVd = LRA

################## end of LRA/UVd preconditioner #################################


##############################################################################
# An Xmat (X-matrix) preconditioner is defined as
#
#   Q = diag(a) + adiag(b)
#
# where adiag means anti-diagonal.
# It's slightly more complicated than a diagonal preconditioner (LRA with rank=0), but may perform better.
#

#@torch.jit.script
def update_precond_Xmat_math_(a, b, v, h, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Update preconditioner Q = diag(a) + adiag(b) with (vector, Hessian-vector product) = (v, h).
    State variables a and b are updated inplace.
    """
    Qh = a*h + b*torch.flip(h, [0])
    aflip, bflip = torch.flip(a, [0]), torch.flip(b, [0])
    invQtv = (aflip*v - bflip*torch.flip(v, [0]))/(a*aflip - b*bflip)
    
    u, v = Qh*Qh, invQtv*invQtv 
    # nablaA = Qh*Qh - invQtv*invQtv
    nablaA = u - v
    nablaB = Qh*torch.flip(Qh, [0]) - invQtv*torch.flip(invQtv, [0])
    q, r = divmod(len(nablaB), 2)
    if r == 1:
        nablaB[q] = 0

    if step_normalizer == '2nd':
        mu = step/(torch.max(u + v) + tiny)
    else:
        mu = step/(torch.maximum(torch.max(torch.abs(nablaA)), torch.max(torch.abs(nablaB))) + tiny)
        
    a.sub_(mu*(nablaA*a + nablaB*bflip))
    b.sub_(mu*(nablaA*b + nablaB*aflip))

#@torch.jit.script
def precond_grad_Xmat_math(a, b, g):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Preconditioning gradient g with Q = diag(a) + adiag(b).
    """
    ab = a * b
    return (a*a + torch.flip(b*b, [0]))*g + (ab + torch.flip(ab, [0]))*torch.flip(g, [0])


class XMat:
    """
    Implements the Xmat preconditioner, Q = diag(a) + adiag(b), as a class.
    Args for initialization:
        params_with_grad: a list of parameters or variables requiring gradients;
        preconditioner_init_scale: initial scale of Q, i.e., Q = preconditioner_init_scale*eye(), with None for automatical setting;
        lr_params: normalized learning rate for parameters in range [0, 1];
        lr_preconditioner: normalized learning rate for preconditioner in range [0, 1];
        momentum: momentum factor in range [0,1);
        grad_clip_max_norm: maximum allowable gradient norm after clipping, None for no clipping;
        preconditioner_update_probability: probability on updating Q, 1 for updating at every step, and 0 for never, i.e., SGD when Q=I;
        step_normalizer: '1st' for normalizing lr_preconditioner with 1st order derivative info, and '2nd' for normalizing with 2nd derivative info;
        exact_hessian_vector_product: True for exact Hessian-vector product via 2nd derivative,
                                    and False for approximate one via the finite difference method;
        preconditioner_type: "Newton" or "whitening", see https://arxiv.org/abs/1809.10232 for the Newton and (empirical) Fisher types.
                                    
    Notes:
        Note 1: The Hessian-vector product can be approximated using the finite difference method by setting
        exact_hessian_vector_product = False when the 2nd derivatives is not available.
        In this case, make sure that the closure produces the same outputs given the same inputs,
        except for numerical errors due to non-deterministic behaviors.
        Random numbers, if any, used inside the closure should be generated starting from the same states, where the rng state can be
        read and set by, e.g., `torch.cuda.get_rng_state' and `torch.cuda.set_rng_state', respectively.
        
        Note 2: Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate.
        This is necessary as the learning rate in PSGD is normalized.

        Note 3: Currently, no support of sparse and mixed-precision gradients.

        Note 4: lr_params, lr_preconditioner, momentum, grad_clip_max_norm, preconditioner_update_probability, step_normalizer, 
        and exact_hessian_vector_product (bool) all can be reset on the fly.
    """
    def __init__(self, params_with_grad, preconditioner_init_scale=None,
                 lr_params=0.01, lr_preconditioner=None, momentum=0.0, 
                 grad_clip_max_norm=None, preconditioner_update_probability=1.0,
                 step_normalizer='2nd',
                 exact_hessian_vector_product: bool = True, preconditioner_type="Newton"):
        # mutable members
        self.lr_params = lr_params
        if lr_preconditioner is None:
            if step_normalizer == '2nd':
                self.lr_preconditioner = 0.1
            else:
                self.lr_preconditioner = 0.01
        else:
            self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        self.exact_hessian_vector_product = exact_hessian_vector_product
        self.step_normalizer = step_normalizer
        # protected members
        params_with_grad = [params_with_grad, ] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]  # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps ** 0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if preconditioner_init_scale is None:
            self._a = None # set it on the fly 
        else:
            self._a = torch.ones(num_params, dtype=dtype, device=device)*preconditioner_init_scale
        self._b = torch.zeros(num_params, dtype=dtype, device=device)
        self._m = None # buffer for momentum 
        self._preconditioner_type = preconditioner_type

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with Xmat preconditioner, i.e.,
        updating the trainable parameters once, and returning what closure returns.
        Args:
            closure (callable): a (stateless) closure that evaluates the function of self._params_with_grad,
                                and returns the loss, or an iterable with the first one being loss.
                                Random numbers, if any, used inside the closure should be generated starting
                                from the same rng state if exact_hessian_vector_product=False and preconditioner_type="Newton".
        """
        if (self._preconditioner_type=="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._a is None)):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self.exact_hessian_vector_product:
                # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(param) for param in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs)
            else:
                # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
            # update preconditioner
            v = torch.cat([torch.flatten(v) for v in vs])
            h = torch.cat([torch.flatten(h) for h in Hvs])
            # initialize self._a if it is None
            if self._a is None: 
                self._a = (torch.sum(v*v)/torch.sum(h*h))**0.25 * torch.ones_like(v)
            # update self._a and self._b
            update_precond_Xmat_math_(self._a, self._b, v, h, self.lr_preconditioner, self.step_normalizer, self._tiny)
            # if self.exact_hessian_vector_product:
            #     update_precond_Xmat_math_(self._a, self._b,
            #                              v, h, step=self.lr_preconditioner, tiny=self._tiny)
            # else:  # compensate the levels of v and h; helpful to reduce numerical errors in half-precision training
            #     update_precond_Xmat_math_(self._a, self._b,
            #                              v/self._delta_param_scale, h/self._delta_param_scale,
            #                              step=self.lr_preconditioner, tiny=self._tiny)
        else:
            # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            vs = None  # no vs and Hvs

        # cat grads
        grad = torch.cat([torch.flatten(g) for g in grads])

        # update preconditioner here if it is the whitening type 
        if (self._preconditioner_type!="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._a is None)):
            if self._a is None:
                self._a = (len(grad)/torch.sum(grad*grad))**0.25 * torch.ones_like(grad)
            # this preconditioner whitens the gradient 
            v = torch.randn_like(grad)
            update_precond_Xmat_math_(self._a, self._b, v, grad, self.lr_preconditioner, self.step_normalizer, self._tiny)
                
        # preconditioned gradients; momentum is optional           
        if self.momentum > 0:
            if self._m is None:
                self._m = (1 - self.momentum)*grad
            else:
                self._m.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
            pre_grad = precond_grad_Xmat_math(self._a, self._b, self._m)
        else:
            self._m = None # clean the buffer when momentum is set to zero again 
            pre_grad = precond_grad_Xmat_math(self._a, self._b, grad)
        
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm / grad_norm, 1.0)

        # update the parameters
        if self.exact_hessian_vector_product or (vs is None) or (self._preconditioner_type!="Newton"):
            delta = lr * pre_grad
        else:  # in this case, do not forget to remove the perturbation on parameters
            delta = lr * pre_grad + v
        # -delta 
        [param.subtract_(delta[j - i:j].view_as(param))
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]

        # return whatever closure returns
        return closure_returns

################## end of Xmat preconditioner #################################


###############################################################################
# The classic Newton-Raphson type preconditioner, but also applicable to indefinite Hessian.
# Clearly, it is applicable only to small scale problems 
#

# @torch.jit.script
def update_precond_newton_math_(Q, invQ, v, h, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Update the classic Newton-Raphson type preconditioner P = Q'*Q with (v, h).
    Similar to but now different from function, 
        update_precond_dense(Q, dxs, dgs, step=0.01, _tiny=1.2e-38)
    """
    a = Q.mm(h)
    if invQ is not None:
        b = invQ.t() @ v
        if step_normalizer == '2nd':
            mu = step/(torch.sum(a*a + b*b) + tiny)
        else: # '1st'
            mu = step*torch.rsqrt(torch.abs(   (torch.sum(a*a))**2
                                              +(torch.sum(b*b))**2
                                            -2*(torch.sum(a*b))**2 ) + tiny) # abs(.) to avoid sqrt(-0.0)
            
        # U = torch.cat([a,  b], 1)     * mu
        # V = torch.cat([a, -b], 1).t() @ Q
        # Q.sub_(U@V)
        # woodbury_identity_(invQ, -U, V)
        
        # U = torch.cat([a,  b], 1)*mu
        # V = torch.cat([a, -b], 1).t()
        # I = torch.eye(2, dtype=U.dtype, device=U.device)
        # Q.sub_(U @ V @ Q)
        # invQ.add_((invQ @ U) @ torch.linalg.solve(I - V@U, V))

        U = torch.cat([a, b], 1) * mu
        V = torch.cat([-a.t() @ Q, v.t()], 0)
        Q.add_(U @ V)
        woodbury_identity_(invQ, U, V)    
    else:
        b = torch.linalg.solve_triangular(Q.t(), v, upper=False)
        grad = torch.triu(a.mm(a.t()) - b.mm(b.t()))
        # grad = triu01(a.mm(a.t()) - b.mm(b.t()))
        if step_normalizer == '2nd':
            mu = step/(torch.sum(a*a + b*b) + tiny)
        else:
            # mu = step/(grad.abs().max() + tiny)
            mu = step/(norm_lower_bound(grad) + tiny)      
            
        Q.sub_(mu*grad.mm(Q))  # use triangular matrix-matrix multiplicationtorch (TRMM) if possible   


class Newton:
    """
    Implements the classic Newton-Raphson type preconditioner for SGD as a class.
    Args for initialization:
        params_with_grad: a list of parameters or variables requiring gradients;
        preconditioner_init_scale: initial scale of Q, i.e., Q = preconditioner_init_scale*eye(), with None for automatical setting;
        lr_params: normalized learning rate for parameters in range [0, 1];
        lr_preconditioner: normalized learning rate for preconditioner in range [0, 1];
        momentum: momentum factor in range [0,1);
        grad_clip_max_norm: maximum allowable gradient norm after clipping, None for no clipping;
        preconditioner_update_probability: probability on updating Q, 1 for updating at every step, and 0 for never, i.e., SGD when Q=I;
        keep_invQ: True for keeping inv(Q) via the rank-2 update of matrix inverse as in BFGS, False for not saving inv(Q); 
        step_normalizer: '1st' for normalizing lr_preconditioner with 1st order derivative info, and '2nd' for normalizing with 2nd derivative info;  
        exact_hessian_vector_product: True for exact Hessian-vector product via 2nd derivative,
                                    and False for approximate one via the finite difference method;
        preconditioner_type: "Newton" or "whitening", see https://arxiv.org/abs/1809.10232 for the Newton and (empirical) Fisher types.
                                    
    Notes:
        Note 1: The Hessian-vector product can be approximated using the finite difference method by setting
        exact_hessian_vector_product = False when the 2nd derivatives is not available.
        In this case, make sure that the closure produces the same outputs given the same inputs,
        except for numerical errors due to non-deterministic behaviors.
        Random numbers, if any, used inside the closure should be generated starting from the same states, where the rng state can be
        read and set by, e.g., `torch.cuda.get_rng_state' and `torch.cuda.set_rng_state', respectively.
        
        Note 2: Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate.
        This is necessary as the learning rate in PSGD is normalized.
        
        Note 3: Currently, no support of sparse and mixed-precision gradients.
        
        Note 4: lr_params, lr_preconditioner, momentum, grad_clip_max_norm, preconditioner_update_probability, step_normalizer, 
        and exact_hessian_vector_product (bool) all can be reset on the fly.
        
        Note 5: Empirical rersults sugguest that keeping inv(Q) via the rank-2 update of matrix inverse as in BFGS is 
        generally numerically safe with single or double precisions, but may not be safe with half precisions.   
        Setting keep_invQ=False disables this feature, and instead backward substitution is used for solving Q^T*x=b. 
    """
    def __init__(self, params_with_grad, preconditioner_init_scale=None,
                 lr_params=0.01, lr_preconditioner=None, momentum=0.0, 
                 grad_clip_max_norm=None, preconditioner_update_probability=1.0,
                 keep_invQ=True, step_normalizer='2nd',
                 exact_hessian_vector_product: bool = True, preconditioner_type="Newton"):
        # mutable members
        self.lr_params = lr_params
        if lr_preconditioner is None:
            if step_normalizer == '2nd':
                self.lr_preconditioner = 0.1
            else:
                self.lr_preconditioner = 0.01
        else:
            self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        self.exact_hessian_vector_product = exact_hessian_vector_product
        self.step_normalizer = step_normalizer 
        # protected members
        params_with_grad = [params_with_grad, ] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]  # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps ** 0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if preconditioner_init_scale is None:
            self._Q = None # initialize Q on the fly
            self._invQ = None 
        else:
            self._Q = torch.eye(num_params, dtype=dtype, device=device)*preconditioner_init_scale
            if keep_invQ:
                self._invQ = torch.eye(num_params, dtype=dtype, device=device)/preconditioner_init_scale
            else:
                self._invQ = None 
        self._m = None # buffer for momentum 
        self._preconditioner_type = preconditioner_type
        self._keep_invQ = keep_invQ 

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with Newton-Raphson type preconditioner, i.e.,
        updating the trainable parameters once, and returning what closure returns.
        Args:
            closure (callable): a (stateless) closure that evaluates the function of self._params_with_grad,
                                and returns the loss, or an iterable with the first one being loss.
                                Random numbers, if any, used inside the closure should be generated starting
                                from the same rng state if exact_hessian_vector_product=False and preconditioner_type="Newton".
        """
        if (self._preconditioner_type=="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._Q is None)):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self.exact_hessian_vector_product:
                # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(param) for param in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs)
            else:
                # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
            # update preconditioner
            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) 
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) 
            # initialize Q if it is None
            if self._Q is None:
                scale = (torch.sum(v*v)/torch.sum(h*h))**0.25
                self._Q = scale * torch.eye(len(v), dtype=v.dtype, device=v.device)
                if self._keep_invQ:
                    self._invQ = torch.eye(len(v), dtype=v.dtype, device=v.device) / scale 
            # then update Q 
            update_precond_newton_math_(self._Q, self._invQ, v, h, self.lr_preconditioner, self.step_normalizer, self._tiny)
            # if self.exact_hessian_vector_product:
            #     update_precond_newton_math_(self._Q,
            #                                 v[:,None], h[:,None], step=self.lr_preconditioner, tiny=self._tiny)
            # else:  # compensate the levels of v and h; helpful to reduce numerical errors in half-precision training
            #     update_precond_newton_math_(self._Q,
            #                                 v[:,None]/self._delta_param_scale, h[:,None]/self._delta_param_scale,
            #                                 step=self.lr_preconditioner, tiny=self._tiny)
        else:
            # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            vs = None  # no vs and Hvs
            
        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) 
        
        # update preconditioner here if it is the whitening type 
        if (self._preconditioner_type!="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._Q is None)):
            if self._Q is None:
                scale = (len(grad)/torch.sum(grad*grad))**0.25
                self._Q = scale * torch.eye(len(grad), dtype=grad.dtype, device=grad.device)
                if self._keep_invQ:
                    self._invQ = torch.eye(len(grad), dtype=grad.dtype, device=grad.device) / scale 
            # this preconditioner whitens the gradient 
            v = torch.randn_like(grad)
            update_precond_newton_math_(self._Q, self._invQ, v, grad, self.lr_preconditioner, self.step_normalizer, self._tiny)

        # preconditioned gradients; momentum is optional             
        if self.momentum > 0:
            if self._m is None:
                self._m = (1 - self.momentum)*grad
            else:
                self._m.mul_(self.momentum).add_(grad, alpha=1 - self.momentum)
            pre_grad = self._Q.t() @ (self._Q @ self._m)
        else:
            self._m = None # clean the buffer when momentum is set to zero again 
            pre_grad = self._Q.t() @ (self._Q @ grad)
        
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm / grad_norm, 1.0)

        # update the parameters
        if self.exact_hessian_vector_product or (vs is None) or (self._preconditioner_type!="Newton"):
            delta = lr * pre_grad 
        else:  # in this case, do not forget to remove the perturbation on parameters
            delta = lr * pre_grad + v
        # -delta 
        [param.subtract_(delta[j - i:j].view_as(param))
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns

################## end of Newton-Raphson preconditioner #################################


##############################################################################
#
# This specific affine group preconditioner is defined as
#
#   Q = sum_i kron(conj(Q2i), Qi1)
#
# where Q1i and Q2i are triangular or diagonal matrices with positive diagonals (https://arxiv.org/pdf/1809.10232.pdf).  
# 


def matrixizer(t):
    """
    It returns triple (f, invf, matrix_shape) for tensor <=> matrix convertion such that
        1) invf(f(t)) = t; 
        2) matrix_shape = f(t).shape 
        3) Preconditioner for matrix f(t) has the minimum size. 
    
    A few examples, 
        1), f(t)=t.reshape([1, 1]) for t = torch.randn([])
        2), f(t)=t.reshape([1, 10]) for t = torch.randn(10)
        3), f(t)=t for t = torch.randn(2, 5)
        4), f(t)=t.reshape(6, 5) for t = torch.randn(2,3,5)
        5), f(t)=t.permute(0,1,3,2,4).reshape(42,55) for t = torch.randn(2,3,5,7,11)
    """
    def prod(arr):
        # prod = lambda arr: 1 if len(arr)==0 else arr[0]*prod(arr[1:])
        result = 1
        for a in arr:
            result *= a
        return result 
    
    def permutations(p0):
        # generate all the permutations of the original one p0 
        if len(p0)==1:
            yield p0
        else:
            for i in range(len(p0)):
                for q in permutations(p0[:i] + p0[i+1:]):
                    yield (p0[i], *q)
    
    # here begins the processing 
    if t.dim() == 2: # t already is a matrix, do nothing  
        return (lambda u: u, lambda v: v, t.shape)
    elif t.dim() < 2: # scalar or vector, simple reshape to matrix  
        mtx_shape = (1, t.numel())
        return (lambda u, shape=mtx_shape: u.reshape(shape),
                lambda v, shape=t.shape:   v.reshape(shape),
                mtx_shape)
    else: # higher order tensor, a little complicated  
        p0, s0 = tuple(range(t.dim())), t.shape # original permutation and shape
        min_precond_size, opt_p,opt_s,opt_i = float('inf'), None,None,None
        for p in permutations(p0):
            s = tuple(s0[j] for j in p)
            for i in range(1, len(p)):  
                if (new_size:=prod(s[:i])**2 + prod(s[i:])**2) < min_precond_size:
                    min_precond_size = new_size
                    opt_p, opt_s, opt_i = p, s, i
            
        if opt_p == p0: # no permutation is needed, just reshaping 
            mtx_shape = (prod(s0[:opt_i]), prod(s0[opt_i:]))
            return (lambda u, shape=mtx_shape: u.reshape(shape),
                    lambda v, shape=s0:        v.reshape(shape),
                    mtx_shape)
        else: # need both permutation and reshaping 
            mtx_shape = (prod(opt_s[:opt_i]), prod(opt_s[opt_i:]))
            q = tuple(pair[1] for pair in sorted([(k,i) for (i,k) in enumerate(opt_p)]))
            return (lambda u, permute=opt_p, shape=mtx_shape: u.permute(permute).reshape(shape),
                    lambda v, permute=q,     shape=opt_s:     v.reshape(shape).permute(permute),
                    mtx_shape)
    
    
def initQ(p, f, scale, max_size, max_skew):
    """
    It initializes preconditioner Q = kron(Q2, Q1) for param p with matrixizer f to scale * I,
    where Q1 or Q2 can reduce to diagonal matrices when either of the following conditions is met,
        1) its size is 1 (no need to save as a matrix);
        2) its size is larger than max_size;
        3) Its relative size compared with the other Q is too large, e.g., size(Q2) > max_skew*size(Q1). 
    """
    s1, s2 = f[2] # f is the returned tuple from matrixizer for param p 
    if s1<2 or s1>max_size or s1>max_skew*s2:
        Q1 = scale**0.5 * torch.ones(s1, dtype=p.dtype, device=p.device) 
    else:
        Q1 = scale**0.5 * torch.eye(s1, dtype=p.dtype, device=p.device) 
        
    if s2<2 or s2>max_size or s2>max_skew*s1:
        Q2 = scale**0.5 * torch.ones(s2, dtype=p.dtype, device=p.device) 
    else:
        Q2 = scale**0.5 * torch.eye(s2, dtype=p.dtype, device=p.device) 
        
    return [Q1, Q2]

# def initQ(size, dtype, device, max_size):
#     """
#     Initialize Q as an identity matrix if its size <= max_size;
#     otherwise a vector of ones. 
#     """
#     if size <= max_size:
#         return torch.eye(size, dtype=dtype, device=device)
#     else:
#         return torch.ones(size, dtype=dtype, device=device) 

#@torch.jit.script
def update_precond_affine_math_(Ql, Qr, dX, dG, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Similar to but now different from function 
        _update_precond_dense_dense(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38)
    Also, it does in-place preconditioner update and support of complex matrices and diagonal matrices.  
    """
    if torch.rand([]) < 0.01:
        max_l = torch.max(torch.abs(Ql)) 
        max_r = torch.max(torch.abs(Qr)) 
        
        rho = torch.sqrt(max_l/max_r)
        Ql.div_(rho)
        Qr.mul_(rho)
        
    
    if Ql.dim()==2:
        if Qr.dim()==2: # Ql.dim()=2 and Qr.dim()=2:
            A = torch.linalg.multi_dot([Ql, dG, Qr.H])
            Bh = torch.linalg.solve_triangular(Ql.H, torch.linalg.solve_triangular(Qr, dX, upper=True, left=False), upper=False) # Bh is B^H 
            
            AhA, BhB = A.H.mm(A), Bh.mm(Bh.H)
            AAh, BBh = A.mm(A.H), Bh.H.mm(Bh)
            # grad1 = torch.triu(A.mm(A.H) - Bh.mm(Bh.H))
            # grad1 = torch.triu(A.mm(A.H) - BhB)
            grad1 = torch.triu(AAh - BhB)
            # grad1 = triu01(A.mm(A.H) - BhB)
            # grad2 = torch.triu(A.H.mm(A) - Bh.H.mm(Bh))
            # grad2 = torch.triu(AhA - Bh.H.mm(Bh))
            grad2 = torch.triu(AhA - BBh)
            # grad2 = triu01(AhA - Bh.H.mm(Bh))
            
            if step_normalizer == '2nd':
                step1 = step/(norm_lower_bound(AAh + BhB) + tiny)
                step2 = step/(norm_lower_bound(AhA + BBh) + tiny)
            else:
                step1 = step/(norm_lower_bound(grad1) + tiny)
                step2 = step/(norm_lower_bound(grad2) + tiny)
                
            Ql.sub_(step1*grad1.mm(Ql)) 
            Qr.sub_(step2*grad2.mm(Qr))
        else: # Ql.dim()=2 and Qr.dim()=1:
            A = Ql.mm(dG*Qr.conj())
            Bh = torch.linalg.solve_triangular(Ql.H, dX/Qr, upper=False) # Bh is B^H
            
            AAh, BhB = A.mm(A.H), Bh.mm(Bh.H)
            AAc, BBc = torch.sum(A*A.conj(), dim=0), torch.sum(Bh*Bh.conj(), dim=0)
            # grad1 = torch.triu(A.mm(A.H) - Bh.mm(Bh.H))
            grad1 = torch.triu(AAh - BhB)
            # grad1 = triu01(AAh - BhB)
            # grad2 = torch.sum(A*A.conj(), dim=0) - torch.sum(Bh*Bh.conj(), dim=0)
            grad2 = AAc - BBc
        
            if step_normalizer == '2nd':
                step1 = step/(norm_lower_bound(AAh + BhB) + tiny)
                step2 = step/(torch.max(torch.real(AAc + BBc)) + tiny)
            else:
                step1 = step/(norm_lower_bound(grad1) + tiny)
                step2 = step/(torch.max(torch.abs(grad2)) + tiny)
                
            Ql.sub_(step1*grad1.mm(Ql)) 
            Qr.sub_(step2*grad2*Qr)
    else: 
        if Qr.dim()==2: # Ql.dim()=1 and Qr.dim()=2:
            A = (Ql[:,None]*dG).mm(Qr.H)
            Bh = torch.linalg.solve_triangular(Qr, dX, upper=True, left=False) / (Ql.conj())[:,None] 
            
            AAc, BBc = torch.sum(A*A.conj(), dim=1), torch.sum(Bh*Bh.conj(), dim=1)
            AhA, BBh = A.H.mm(A), Bh.H.mm(Bh)
            # grad1 = torch.sum(A*A.conj(), dim=1) - torch.sum(Bh*Bh.conj(), dim=1)
            grad1 = AAc - BBc
            # grad2 = torch.triu(A.H.mm(A) - Bh.H.mm(Bh))
            grad2 = torch.triu(AhA - BBh)
            # grad2 = triu01(AhA - BBh)
        
            if step_normalizer == '2nd':
                step1 = step/(torch.max(torch.real(AAc + BBc)) + tiny)
                step2 = step/(norm_lower_bound(AhA + BBh) + tiny)
            else:
                step1 = step/(torch.max(torch.abs(grad1)) + tiny)
                step2 = step/(norm_lower_bound(grad2) + tiny)
                
            Ql.sub_(step1*grad1*Ql) 
            Qr.sub_(step2*grad2.mm(Qr))
        else: # Ql.dim()=1 and Qr.dim()=1:
            A = Ql[:,None] * dG * Qr.conj()
            Bh = dX / Qr / (Ql.conj())[:,None] 
            
            AAc1, BBc1 = torch.sum(A*A.conj(), dim=1), torch.sum(Bh*Bh.conj(), dim=1)
            AAc2, BBc2 = torch.sum(A*A.conj(), dim=0), torch.sum(Bh*Bh.conj(), dim=0)
            # grad1 = torch.sum(A*A.conj(), dim=1) - torch.sum(Bh*Bh.conj(), dim=1)
            grad1 = AAc1 - BBc1
            # grad2 = torch.sum(A*A.conj(), dim=0) - torch.sum(Bh*Bh.conj(), dim=0)
            grad2 = AAc2 - BBc2
        
            if step_normalizer == '2nd':
                step1 = step/(torch.max(torch.real(AAc1 + BBc1)) + tiny)
                step2 = step/(torch.max(torch.real(AAc2 + BBc2)) + tiny)
            else:
                step1 = step/(torch.max(torch.abs(grad1)) + tiny)
                step2 = step/(torch.max(torch.abs(grad2)) + tiny)
                
            Ql.sub_(step1*grad1*Ql) 
            Qr.sub_(step2*grad2*Qr)
    

#@torch.jit.script
def update_precond_affine_dropv_math_(Ql, Qr, dG, step, step_normalizer, tiny):
    # type: (Tensor, Tensor, Tensor, float, str, float) -> None
    """
    Similar to fuction 
        update_precond_affine_math_,
    but exclusively for the Affine gradient whitening preconditioner so that we can integrate out dummy variable v if desirable. 
    """
    def balance():      
        if torch.rand([]) < 0.01:
            max_l = torch.max(torch.abs(Ql)) 
            max_r = torch.max(torch.abs(Qr)) 
            
            rho = torch.sqrt(max_l/max_r)
            Ql.div_(rho)
            Qr.mul_(rho)
        
        
    if Ql.dim()==1 and Qr.dim()==1:
        # drop v when both dims use diagonal preconditioners 
        A = Ql[:,None] * dG * Qr.conj()
        invQQl, invQQr = 1/(Ql*Ql.conj()), 1/(Qr*Qr.conj())
        
        AAc1, BBc1 = torch.sum(A*A.conj(), dim=1), torch.sum(invQQr) * invQQl 
        AAc2, BBc2 = torch.sum(A*A.conj(), dim=0), torch.sum(invQQl) * invQQr 
        grad1 = AAc1 - BBc1
        grad2 = AAc2 - BBc2
    
        if step_normalizer == '2nd':
            step1 = step/(torch.max(torch.real(AAc1 + BBc1)) + tiny)
            step2 = step/(torch.max(torch.real(AAc2 + BBc2)) + tiny)
        else:
            step1 = step/(torch.max(torch.abs(grad1)) + tiny)
            step2 = step/(torch.max(torch.abs(grad2)) + tiny)
            
        Ql.sub_(step1*grad1*Ql) 
        Qr.sub_(step2*grad2*Qr)
        balance()
    elif Ql.dim()==1 and Ql.shape[0]>=Qr.shape[0]: # Qr.dim() == 2 in this case 
        # drop v when left is diagonal, right is dense, and gradient is a tall matrix
        A = (Ql[:,None]*dG).mm(Qr.H)
        invQQl = 1/(Ql*Ql.conj())
        invQr = torch.linalg.solve_triangular(Qr, torch.eye(Qr.shape[0], dtype=Qr.dtype, device=Qr.device), upper=True)
        invQQr = invQr.H @ invQr
        
        AAc, BBc = torch.sum(A*A.conj(), dim=1), torch.trace(invQQr) * invQQl 
        AhA, BBh = A.H.mm(A), torch.sum(invQQl) * invQQr 
        grad1 = AAc - BBc
        grad2 = torch.triu(AhA - BBh)
    
        if step_normalizer == '2nd':
            step1 = step/(torch.max(torch.real(AAc + BBc)) + tiny)
            step2 = step/(norm_lower_bound(AhA + BBh) + tiny)
        else:
            step1 = step/(torch.max(torch.abs(grad1)) + tiny)
            step2 = step/(norm_lower_bound(grad2) + tiny)
            
        Ql.sub_(step1*grad1*Ql) 
        Qr.sub_(step2*grad2.mm(Qr))   
        balance()
    elif Qr.dim()==1 and Qr.shape[0]>=Ql.shape[0]: # Ql.dim() == 2 in this case 
        # drop v when right is diagonal, left is dense, and gradient is a short matrix 
        A = Ql.mm(dG*Qr.conj())
        invQl = torch.linalg.solve_triangular(Ql, torch.eye(Ql.shape[0], dtype=Ql.dtype, device=Ql.device), upper=True)
        invQQl = invQl.H @ invQl
        invQQr = 1/(Qr*Qr.conj())
        
        AAh, BhB = A.mm(A.H), torch.sum(invQQr) * invQQl 
        AAc, BBc = torch.sum(A*A.conj(), dim=0), torch.trace(invQQl) * invQQr 
        grad1 = torch.triu(AAh - BhB)
        grad2 = AAc - BBc
    
        if step_normalizer == '2nd':
            step1 = step/(norm_lower_bound(AAh + BhB) + tiny)
            step2 = step/(torch.max(torch.real(AAc + BBc)) + tiny)
        else:
            step1 = step/(norm_lower_bound(grad1) + tiny)
            step2 = step/(torch.max(torch.abs(grad2)) + tiny)
            
        Ql.sub_(step1*grad1.mm(Ql)) 
        Qr.sub_(step2*grad2*Qr)
        balance()
    else:
        # keeping v as an auxiliary variable could save computations (tradeoff of performance, similar to Hutchinson’s trick) when
        #   1) gradient is a tall matrix, but left side is a dense preconditioner, right side is diagonal
        #   2) gradient is a short matrix, but left side is a diagonal preconditioner, right side is dense
        #   3) both sides use dense preconditioner, but gradient is skewed (no saving for square shape gradient)
        update_precond_affine_math_(Ql, Qr, torch.randn_like(dG), dG, step, step_normalizer, tiny)


#@torch.jit.script
def precond_grad_affine_math(Ql, Qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Basically a copy of function 
        _precond_grad_dense_dense(Ql, Qr, Grad)
    but with support of complex matrices and diagonal matrices 
    """
    if Ql.dim()==2:
        if Qr.dim()==2: # Ql.dim()=2 and Qr.dim()=2:
            return torch.linalg.multi_dot([Ql.H, Ql, Grad, Qr.H, Qr])
        else: # Ql.dim()=2 and Qr.dim()=1:
            return torch.linalg.multi_dot([Ql.H, Ql, Grad*(Qr*Qr.conj())])
    else:
        if Qr.dim()==2: # Ql.dim()=1 and Qr.dim()=2:
            return torch.linalg.multi_dot([(Ql*Ql.conj())[:,None] * Grad, Qr.H, Qr])
        else: # Ql.dim()=1 and Qr.dim()=1:
            return (Ql*Ql.conj())[:,None] * Grad * (Qr*Qr.conj())


class Affine:
    """
    Implements the affine group preconditioner, Q = sum_i kron(conj(Q2i), Q1i), as a class.

    Args for initialization:
        params_with_grad: a list of real or complex matrix or tensor parameters requiring gradients;
        preconditioner_max_size: Q2i or Q1i reduces to a diagonal matrix if its size is larger than preconditioner_max_size, otherwise a triangular matrix;
        preconditioner_max_skew: for example, Q2 reduces to a diagonal matrix if size(Q2)>preconditioner_max_skew*size(Q1), otherwise a triangular matrix;
        preconditioner_init_scale: initial scale of Q, i.e., Q = preconditioner_init_scale*eye(), with None for automatical setting;
        lr_params: normalized learning rate for parameters in range [0, 1];
        lr_preconditioner: normalized learning rate for preconditioner in range [0, 1];
        momentum: momentum factor in range [0,1);
        grad_clip_max_norm: maximum allowable gradient norm after clipping, None for no clipping;
        preconditioner_update_probability: probability on updating Q, 1 for updating at every step, and 0 for never, i.e., SGD when Q=I;
        step_normalizer: '1st' for normalizing lr_preconditioner with 1st order derivative info, and '2nd' for normalizing with 2nd derivative info; 
        exact_hessian_vector_product: True for exact Hessian-vector product via 2nd derivative,
                                    and False for approximate one via the finite difference method;
        preconditioner_type: "Newton" or "whitening", see https://arxiv.org/abs/1809.10232 for the Newton and (empirical) Fisher types.  

    Notes:
        Note 1: All the preconditioner matrices are triangular matrices if preconditioner_max_size=preconditioner_max_skew=inf (max memory consumption), 
        and diagonal matrices if either preconditioner_max_size or preconditioner_max_skew is zero (least memory consumption). 
        We can control the memory consumption by setting preconditioner_max_size and preconditioner_max_skew properly.
        
        Note 2: Affine preconditioners are not black box ones. It is the user's responsibility to reparameterize the model parameters as a list of matrices.
        Otherwise, non-2D tensor parameters are reshaped to matrices as specified by the function matrixizer. 
            
        Note 3: The Hessian-vector product can be approximated using the finite difference method by setting 
        exact_hessian_vector_product = False when the 2nd derivatives is not available.
        In this case, make sure that the closure produces the same outputs given the same inputs, 
        except for numerical errors due to non-deterministic behaviors.
        Random numbers, if any, used inside the closure should be generated starting from the same state, where the rng state can be
        read and set by, e.g., `torch.cuda.get_rng_state' and `torch.cuda.set_rng_state', respectively.
        
        Note 4: Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate.
        This is necessary as the learning rate in PSGD is normalized.    
        
        Note 5: lr_params, lr_preconditioner, momentum, grad_clip_max_norm, preconditioner_update_probability, and 
        exact_hessian_vector_product (bool) all can be reset on the fly. 
        
        Note 6: The matrices and tensor parameters to be optimized can be of different data types (real or complex, single or double, etc.). 
    """
    def __init__(self,  params_with_grad, preconditioner_max_size=torch.inf, preconditioner_max_skew=torch.inf, preconditioner_init_scale=None,
                        lr_params=0.01, lr_preconditioner=None, momentum=0.0,
                        grad_clip_max_norm=None, preconditioner_update_probability=1.0,
                        step_normalizer='2nd',
                        exact_hessian_vector_product:bool=True, preconditioner_type="Newton"):
        # mutable members
        self.lr_params = lr_params
        if lr_preconditioner is None:
            if step_normalizer == '2nd':
                self.lr_preconditioner = 0.1
            else:
                self.lr_preconditioner = 0.01
        else:
            self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        self.exact_hessian_vector_product = exact_hessian_vector_product
        self.step_normalizer = step_normalizer
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        self._tiny = max([torch.finfo(p.dtype).tiny for p in self._params_with_grad])
        self._delta_param_scale = (max([torch.finfo(p.dtype).eps for p in self._params_with_grad])) ** 0.5
        self._matrixizers = tuple(matrixizer(p) for p in self._params_with_grad)
        if preconditioner_init_scale is None:
            self._Qs = None # initialize on the fly 
        else:
            self._Qs = [initQ(p,f, preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for (f,p) in zip(self._matrixizers, self._params_with_grad)]
            # self._Qs = [[preconditioner_init_scale**0.5 * initQ(p.shape[0], p.dtype, p.device, preconditioner_max_size), 
            #              preconditioner_init_scale**0.5 * initQ(p.shape[1], p.dtype, p.device, preconditioner_max_size)] for p in self._params_with_grad]
        self._ms = None # momentum buffers 
        self._preconditioner_type = preconditioner_type
        # echo matrixizer info 
        for i, p in enumerate(self._params_with_grad):
            if p.dim() != 2:
                print(f"FYI: expect 2D params; got {p.dim()}D with shape {p.shape} for the {i}th param; will reshape to {self._matrixizers[i][2]}")
                # raise ValueError(f"expect 2D params (got {p.dim()}D param)")   

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the affine group preconditioner, i.e., 
        updating the trainable parameters once, and returning what closure returns.

        Args:
            closure (callable): a (stateless) closure that evaluates the function of self._params_with_grad,
                                and returns the loss, or an iterable with the first one being loss.
                                Random numbers, if any, used inside the closure should be generated starting 
                                from the same rng state if exact_hessian_vector_product=False and preconditioner_type="Newton". 
        """
        if (self._preconditioner_type=="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._Qs is None)):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self.exact_hessian_vector_product:
                # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(p) for p in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs) # this line also works for complex matrices 
            else:
                # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                vs = [self._delta_param_scale * torch.randn_like(p) for p in self._params_with_grad]
                [p.add_(v) for (p, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
            # update preconditioner 
            # initialize Qs if it is None 
            if self._Qs is None:
                self._Qs = [initQ(v,f, (torch.sum(v*v.conj())/torch.sum(h*h.conj()))**0.25, self._preconditioner_max_size, self._preconditioner_max_skew) for (f,v,h) in zip(self._matrixizers, vs, Hvs)]
                # self._Qs = [[(torch.sum(v*v.conj())/torch.sum(h*h.conj()))**0.25 * initQ(v.shape[0], v.dtype, v.device, self._preconditioner_max_size), 
                #                                                                    initQ(v.shape[1], v.dtype, v.device, self._preconditioner_max_size)] for (v, h) in zip(vs, Hvs)]
            # update self._Qs
            [update_precond_affine_math_(Qlr[0], Qlr[1], f[0](v), f[0](h), self.lr_preconditioner, self.step_normalizer, self._tiny) for (Qlr, f,v,h) in zip(self._Qs, self._matrixizers, vs, Hvs)]
            # [update_precond_affine_math_(Qlr[0], Qlr[1], v, h, self.lr_preconditioner, self.step_normalizer, self._tiny) for (Qlr, v, h) in zip(self._Qs, vs, Hvs)]
        else:
            # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            vs = None # no vs and Hvs
        
        # update preconditioner here if it is the whitening type 
        if (self._preconditioner_type!="Newton") and ((torch.rand([]) < self.preconditioner_update_probability) or (self._Qs is None)):
            if self._Qs is None:
                self._Qs = [initQ(g,f, (torch.numel(g)/torch.sum(g*g.conj()))**0.25, self._preconditioner_max_size, self._preconditioner_max_skew) for (f,g) in zip(self._matrixizers, grads)]
                # self._Qs = [[(torch.numel(g)/torch.sum(g*g.conj()))**0.25 * initQ(g.shape[0], g.dtype, g.device, self._preconditioner_max_size), 
                #                                                             initQ(g.shape[1], g.dtype, g.device, self._preconditioner_max_size)] for g in grads]
            # update the preconditioner whitening the gradients 
            [update_precond_affine_dropv_math_(Qlr[0], Qlr[1], f[0](g), self.lr_preconditioner, self.step_normalizer, self._tiny) for (Qlr, f,g) in zip(self._Qs, self._matrixizers, grads)]
            # [update_precond_affine_math_(Qlr[0], Qlr[1], f[0](torch.randn_like(g)), f[0](g), self.lr_preconditioner, self.step_normalizer, self._tiny) for (Qlr, f,g) in zip(self._Qs, self._matrixizers, grads)]
            # [update_precond_affine_math_(Qlr[0], Qlr[1], torch.randn_like(g), g, self.lr_preconditioner, self.step_normalizer, self._tiny) for (Qlr, g) in zip(self._Qs, grads)]

        # preconditioned gradients; momentum is optional      
        if self.momentum > 0:
            if self._ms is None:
                self._ms = [(1 - self.momentum)*g for g in grads]
            else:
                [m.mul_(self.momentum).add_(g, alpha=1 - self.momentum) for (m, g) in zip(self._ms, grads)]
            pre_grads = [precond_grad_affine_math(Qlr[0], Qlr[1], f[0](m)) for (Qlr, f,m) in zip(self._Qs, self._matrixizers, self._ms)]
            # pre_grads = [precond_grad_affine_math(Qlr[0], Qlr[1], m) for (Qlr, m) in zip(self._Qs, self._ms)]
        else:
            self._ms = None # clean the buffer when momentum is set to zero 
            pre_grads = [precond_grad_affine_math(Qlr[0], Qlr[1], f[0](g)) for (Qlr, f,g) in zip(self._Qs, self._matrixizers, grads)]
            # pre_grads = [precond_grad_affine_math(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(self._Qs, grads)]
            
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.sqrt(torch.abs(sum([torch.sum(g*g.conj()) for g in pre_grads]))) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # update the parameters
        if self.exact_hessian_vector_product or (vs is None) or (self._preconditioner_type!="Newton"):
            [param.subtract_(lr*f[1](g)) for (param, f,g) in zip(self._params_with_grad, self._matrixizers, pre_grads)]
            # [param.subtract_(lr*g) for (param, g) in zip(self._params_with_grad, pre_grads)]
        else: # in this case, do not forget to remove the perturbation on parameters
            [param.subtract_(lr*f[1](g) + v) for (param, f,g,v) in zip(self._params_with_grad, self._matrixizers, pre_grads, vs)]
            # [param.subtract_(lr*g + v) for (param, g, v) in zip(self._params_with_grad, pre_grads, vs)]        
        
        # return whatever closure returns
        return closure_returns

################## end of the Affine preconditioner #################################
