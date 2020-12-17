"""Created in May, 2018
Pytorch functions for preconditioned SGD
@author: XILIN LI, lixilinx@gmail.com

Updated in Dec, 2020: 
Wrapped Kronecker product preconditioner for easy use: the code will select the proper Kronecker product  
preconditioner based on the formats of input left and right preconditioners
"""

import torch


###############################################################################
def update_precond_dense(Q, dxs, dgs, step=0.01, _tiny=1.2e-38):
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
    b = torch.triangular_solve(dx, Q, upper=True, transpose=True)[0]

    grad = torch.triu(a.mm(a.t()) - b.mm(b.t()))
    step0 = step/(grad.abs().max() + _tiny)        
        
    return Q - step0*grad.mm(Q)


def precond_grad_dense(Q, grads):
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
            return update_precond_dense_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p==2: # (dense, normalization) format
            return update_precond_norm_dense(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p==1: # (dense, scaling) format
            return update_precond_dense_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return update_precond_norm_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p==1: # (normalization, scaling) format
            return update_precond_norm_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return update_precond_dense_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p==2: # (scaling, normalization) format
            return update_precond_norm_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
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
            return precond_grad_dense_dense(Ql, Qr, Grad)
        elif p==2: # (dense, normalization) format
            return precond_grad_norm_dense(Qr, Ql, Grad.t()).t()
        elif p==1: # (dense, scaling) format
            return precond_grad_dense_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==2: # left is normalization
        if p==q: # (normalization, dense) format
            return precond_grad_norm_dense(Ql, Qr, Grad)
        elif p==1: # (normalization, scaling) format
            return precond_grad_norm_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m==1: # left is scaling
        if p==q: # (scaling, dense) format
            return precond_grad_dense_scale(Qr, Ql, Grad.t()).t()
        elif p==2: # (scaling, normalization) format
            return precond_grad_norm_scale(Qr, Ql, Grad.t()).t()
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    else:
        raise Exception('Unknown Kronecker product preconditioner')
        

###############################################################################
def update_precond_dense_dense(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
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
    Ql = Ql/rho
    Qr = rho*Qr
    
    A = Ql.mm( dG.mm( Qr.t() ) )
    Bt = torch.triangular_solve((torch.triangular_solve(dX.t(), Qr, upper=True, transpose=True))[0].t(), 
                     Ql, upper=True, transpose=True)[0]
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    
    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), Qr - step2*grad2.mm(Qr)
    

def precond_grad_dense_dense(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    return torch.chain_matmul(Ql.t(), Ql, Grad, Qr.t(), Qr)
    

###############################################################################
# (normalization, dense) format Kronecker product preconditioner
def update_precond_norm_dense(ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
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
    ql = ql/rho
    Qr = rho*Qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t()*dG
    A = A + ql[1:].t().mm( dG[-1:] ) # Ql*dG 
    A = A.mm(Qr.t())
    
    Bt = (1.0/ql[0:1].t())*dX
    Bt[-1:] = Bt[-1:] - (ql[1:]/(ql[0:1]*ql[0,-1])).mm(dX)
    Bt = torch.triangular_solve(Bt.t(), Qr, upper=True, transpose=True)[0].t()
    
    grad1_diag = torch.sum(A*A, dim=1) - torch.sum(Bt*Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t()) 
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])  

    step1 = step/(max(torch.max(torch.abs(grad1_diag)), 
                      torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    
    return torch.stack((new_ql0, new_ql1)), Qr - step2*grad2.mm(Qr)


def precond_grad_norm_dense(ql, Qr, Grad):
    """
    return preconditioned gradient using (normalization, dense) Kronecker product preconditioner 
    Suppose Grad has shape (M, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    Qr: shape (N, N), Cholesky factor of right preconditioner
    Grad: (matrix) gradient
    """
    preG = ql[0:1].t()*Grad
    preG = preG + ql[1:].t().mm(Grad[-1:]) # Ql*Grad 
    preG = torch.chain_matmul(preG, Qr.t(), Qr)
    add_last_row = ql[1:].mm(preG) # use it to modify the last row
    preG = ql[0:1].t()*preG
    preG[-1:] = preG[-1:] + add_last_row
    
    return preG


###############################################################################
# (normalization, scaling) Kronecker product preconditioner 
# the left one is a normalization preconditioner; the right one is a scaling preconditioner
def update_precond_norm_scale(ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
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
    ql = ql/rho
    qr = rho*qr
    
    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t()*dG
    A = A + ql[1:].t().mm( dG[-1:] ) # Ql*dG 
    A = A*qr # Ql*dG*Qr 
    
    Bt = (1.0/ql[0:1].t())*dX
    Bt[-1:] = Bt[-1:] - (ql[1:]/(ql[0:1]*ql[0,-1])).mm(dX)
    Bt = Bt*(1.0/qr) # Ql^(-T)*dX*Qr^(-1) 
    
    grad1_diag = torch.sum(A*A, dim=1) - torch.sum(Bt*Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t()) 
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])  

    step1 = step/(max(torch.max(torch.abs(grad1_diag)), 
                      torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1*grad1_diag*ql[0]
    new_ql1 = ql[1] - step1*(grad1_diag*ql[1] + ql[0,-1]*grad1_bias)
    
    grad2 = torch.sum(A*A, dim=0, keepdim=True) - torch.sum(Bt*Bt, dim=0, keepdim=True)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
    
    return torch.stack((new_ql0, new_ql1)), qr - step2*grad2*qr


def precond_grad_norm_scale(ql, qr, Grad):
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
    preG = ql[0:1].t()*Grad
    preG = preG + ql[1:].t().mm(Grad[-1:]) # Ql*Grad 
    preG = preG*(qr*qr) # Ql*Grad*Qr^T*Qr
    add_last_row = ql[1:].mm(preG) # use it to modify the last row
    preG = ql[0:1].t()*preG
    preG[-1:] = preG[-1:] + add_last_row
    
    return preG


###############################################################################
def update_precond_dense_scale(Ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
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
    Ql = Ql/rho
    qr = rho*qr
    
    A = Ql.mm( dG*qr )
    Bt = torch.triangular_solve(dX/qr, Ql, upper=True, transpose=True)[0]
    
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.sum(A*A, dim=0, keepdim=True) - torch.sum(Bt*Bt, dim=0, keepdim=True)
    
    step1 = step/(torch.max(torch.abs(grad1)) + _tiny)
    step2 = step/(torch.max(torch.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.mm(Ql), qr - step2*grad2*qr
    

def precond_grad_dense_scale(Ql, qr, Grad):
    """
    return preconditioned gradient using (dense, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    Ql: shape (M, M), (left side) Cholesky factor of preconditioner
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    return torch.chain_matmul(Ql.t(), Ql, Grad*(qr*qr))



###############################################################################                        
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=0.01, _tiny=1.2e-38):
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
    max_l = max(torch.max(torch.diag(L12)), torch.max(l3))
    max_u = max(torch.max(torch.diag(U12)), torch.max(u3))
    rho = torch.sqrt(max_l/max_u)
    L12 = L12/rho
    l3 = l3/rho
    U12 = rho*U12
    u3 = rho*u3
    
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
    iUtx1 = torch.triangular_solve(dx[:r], U1, upper=True, transpose=True)[0]
    iUtx2 = (dx[r:] - U2.t().mm(iUtx1))/u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2/l3
    iQtx1 = torch.triangular_solve(iUtx1 - L2.t().mm(iQtx2), L1, upper=False, transpose=True)[0]
    # L^T*Q*dg
    LtQg1 = L1.t().mm(Qg1) + L2.t().mm(Qg2)
    LtQg2 = l3*Qg2
    # P*dg
    Pg1 = U1.t().mm(LtQg1)
    Pg2 = U2.t().mm(LtQg1) + u3*LtQg2
    # inv(L)*inv(Q^T)*dx
    iLiQtx1 = torch.triangular_solve(iQtx1, L1, upper=False)[0]
    iLiQtx2 = (iQtx2 - L2.mm(iLiQtx1))/l3
    # inv(P)*dx
    iPx2 = iLiQtx2/u3
    iPx1 = torch.triangular_solve(iLiQtx1 - U2.mm(iPx2), U1, upper=True)[0]
    
    # update L
    grad1 = Qg1.mm(Qg1.t()) - iQtx1.mm(iQtx1.t())
    grad1 = torch.tril(grad1)
    grad2 = Qg2.mm(Qg1.t()) - iQtx2.mm(iQtx1.t())
    grad3 = Qg2*Qg2 - iQtx2*iQtx2
    max_abs_grad = torch.max(torch.abs(grad1))
    max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad2)))
    max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newL1 = L1 - step0*grad1.mm(L1)
    newL2 = L2 - step0*grad2.mm(L1) - step0*grad3*L2
    newl3 = l3 - step0*grad3*l3

    # update U
    grad1 = Pg1.mm(dg[:r].t()) - dx[:r].mm(iPx1.t())
    grad1 = torch.triu(grad1)
    grad2 = Pg1.mm(dg[r:].t()) - dx[:r].mm(iPx2.t())
    grad3 = Pg2*dg[r:] - dx[r:]*iPx2
    max_abs_grad = torch.max(torch.abs(grad1))
    max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad2)))
    max_abs_grad = max(max_abs_grad, torch.max(torch.abs(grad3)))
    step0 = step/(max_abs_grad + _tiny)
    newU1 = U1 - U1.mm(step0*grad1)
    newU2 = U2 - U1.mm(step0*grad2) - step0*grad3.t()*U2
    newu3 = u3 - step0*grad3*u3

    return torch.cat([newL1, newL2], dim=0), newl3, torch.cat([newU1, newU2], dim=1), newu3


def precond_grad_splu(L12, l3, U12, u3, grads):
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