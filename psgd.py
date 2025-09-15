"""
The new PSGD-Kron Newton/Whitening preconditioners support five kinds of local coordinates for updating Q: 

    QUAD): It's a specific form for updating Q to ensure that Q > 0 (thus Q is symmetric/Hermitian).   
    This is one of the recommended choices for fitting Q. 

    QEQ):    dQ = Q * mathcal{E} * Q
    This leads to another simple way for updating Q (Q is in the general linear group).
    It's another recommended choice for fitting Q.  

    Q0p5EQ1p5): dQ = Q^0.5 * mathcal{E} * Q^1.5
    One more recommended choice for fitting Q. 
    An online orthogonal Procrustes problem solver is used to keep Q approximately SPD.  

    EQ):    dQ = mathcal{E} * Q
    This choice recovers the old PSGD way for updating Q in Lie groups (Q is triangular). 
    Its main drawback is that triangualr solvers are required for updating Q.  

    QEP):   dQ = Q * mathcal{E} * P
    This last choice works very well if it does. Q is in the general linear group.  
    But, one drawback is that Q might get stuck around ill-conditioned matrices (not strongly convex). 

The QUAD formulae can be used to update P directly (see older commit 0fc33cd). 
I call this choice QUAD4P. It still is a good choice for optimization with single precision. 
Unlike QUAD, QUAD4P does not work well with half precision. Use it with caution.  

The PSGD-LRA Newton/Whitening preconditioners still adopt local coordinate dQ = mathcal{E} * Q, 
and needs a small linear solver to update the preconditioner.

I also keep the PSGD dense matrix Newton-type preconditioner here to illustrate the math. 
It supports all the five methods for updating Q, 
and can be a good alternative to the BFGS like quasi-Newton optimizers as no line search is required. 

Xi-Lin Li, lixilinx@gmail.com; last updated in Sept., 2025. 
Main refs: https://arxiv.org/abs/1512.04202; https://arxiv.org/abs/2402.11858. 
"""

import opt_einsum
import torch


def norm_lower_bound_spd(A):
    """
    Returns a cheap lower bound for the spectral norm of a symmetric positive definite matrix A.
    """
    max_abs = A.diagonal().real.amax() # used to normalize A to avoid numerical under/over-flow
    if max_abs > torch.finfo(max_abs.dtype).smallest_normal: # to avoid inf due to 1/subnormal or 1/0
        A = A/max_abs
        j = torch.argmax(torch.real(torch.sum(A * A.conj(), dim=1)))
        x = A[j] @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A)
    else: # virtually A=0
        return max_abs 


def norm_lower_bound_skh(A):
    """
    Returns a cheap lower bound for the spectral norm of a skew-Hermitian matrix A. 
    """
    max_abs = A.abs().amax() # used to normalize A to avoid numerical under/over-flow
    if max_abs > torch.finfo(max_abs.dtype).smallest_normal: # to avoid inf due to 1/subnormal or 1/0
        A = A/max_abs
        j = torch.argmax(torch.real(torch.sum(A * A.conj(), dim=1)))
        x = A[j] @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A)
    else: # virtually A=0
        return max_abs 
    

def lift2single(x):
    # lift half or lower precision to single precision; leave single or higher precision unchanged  
    return x.to(torch.float32) if torch.finfo(x.dtype).eps > 1e-6 else x
    

def procrustes_step(Q, max_step_size=0.2):
    """
    A in-place (update Q directly) online solver for the orthogonal Procrustes problem,
        min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator and a ||R|| < 1. 

    Note that such rotations do not include reflections, and thus cannot make real Q SPD if det(Q) < 0. 
    """
    R = Q.H - Q 
    max_abs = R.abs().amax()
    if max_abs > torch.finfo(max_abs.dtype).smallest_normal: # to avoid inf due to 1/subnormal or 1/0
        R /= max_abs # normalize R as typically it's too small 
        RQ = R @ Q
        tr_RQ = RQ.diagonal().real.sum() # torch.trace not implemented for bfloat16, so sum(diag())  
        if tr_RQ > 0: # otherwise tr_RQ = 0 and thus Q is already Hermitian  
            # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal a
            a = max_step_size / norm_lower_bound_skh(R)
            RRQ = R @ RQ
            tr_RRQ = RRQ.diagonal().real.sum() 
            if tr_RRQ < 0: # the max step size could over-shoot in this case 
                a = min(a, -tr_RQ / tr_RRQ) 
            Q.add_(a * (RQ + 0.5 * a * RRQ))


#############       Begin of PSGD Kronecker product preconditioners       #############         


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0, dQ="QEQ"):
    """
    For a scalar or tensor t, we initialize its states (preconditioner Q and Lipschitz smoothness constant L), 
    and reusable contraction expressions for updating Q and preconditioning gradient.
    
    1, The preconditioner Q is initialized to 
        Q = Scale * I = Scale * kron(eye(t.shape[0]), eye(t.shape[1]), ...)
       where the eye(.) may be replaced with diag(ones(.)) if that dim is too large, determined by max_size and max_skew.
       
       The Lipschitz smoothness constant L for Q is initialized to zero. 
       
    2, A series of enisum contract expressions. The following subscript examples are for a 5th order tensor.  
        2.1, exprP is the expression for applying the Preconditioner on the gradient, e.g.,
                'aA,bB,cC,dD,eE,aα,bβ,cγ,dδ,eε,αβγδε->ABCDE'
        2.2, the i-th expression of exprGs is for the contraction of two tensors that only keeps the i-th dim, e.g.,
                'abCde,abγde->Cγ'
            for i=2. It's useful for Gradient calculation.  
        2.3, exprA is the expression for applying All the factors of Q on a tensor, e.g.,
                'aA,bB,cC,dD,eE,ABCDE->abcde' 
        2.4, the i-th expression of exprQs is the expression for applying the i-th factor of Q on a tensor, e.g., 
                'Cγ,abγde->abCde'
            for i=2. 

        Please check https://drive.google.com/file/d/1CEEq7A3_l8EcPEDa_sYtqr5aMLVeZWL7/view?usp=drive_link for notations and derivations. 
    """
    if dQ == "QUAD4P": # the only case that we fit P directly; so square Scale 
        Scale = Scale ** 2 
    shape = t.shape 
    if len(shape)==0: # scalar 
        Q = [Scale * torch.ones_like(t),]
        L = [lift2single(torch.zeros_like(t.real)),]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape) 
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape),]
        exprQs = [opt_einsum.contract_expression(",->", Q[0].shape, t.shape),]
    else: # tensor 
        if len(shape) > 26:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; einsum runs out of letters; replace 26 with larger numbers.")   
            
        scale = Scale ** (1/len(shape)) 
    
        Q, L = [], []
        exprGs, exprQs = [], []
        piece1A, piece2A, piece3A = [], "", "" # used for getting the subscripts for exprA
        piece1P, piece2P, piece3P, piece4P = [], [], "", "" # used for getting the subscripts for exprP
        for i, size in enumerate(shape):
            L.append(lift2single(torch.zeros([], dtype=t.real.dtype, device=t.device)))
            if size <= 1 or size > max_size or size**2 > max_skew * t.numel():
                # use diagonal matrix as preconditioner for this dim 
                Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
                
                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i+26)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i+26) + "," + piece1 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
            else: # use matrix preconditioner for this dim 
                Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = opt_einsum.get_symbol(i), opt_einsum.get_symbol(i + 26), opt_einsum.get_symbol(i + 805)
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                piece2 = "".join([opt_einsum.get_symbol(i+805) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + opt_einsum.get_symbol(i+26) + opt_einsum.get_symbol(i+805)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i+26) + opt_einsum.get_symbol(i+805) + "," + piece2 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
        
        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], t.shape)

        subscripts = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        exprP = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape)
    
    exprGs, exprQs = tuple(exprGs), tuple(exprQs)
    if dQ == "QEP": 
        return [[Q, L], (exprP, exprGs, exprQs)]
    elif dQ == "EQ": 
        return [[Q, L], (exprP, exprGs, exprA)]
    elif (dQ == "QEQ") or (dQ == "QUAD") or (dQ == "Q0p5EQ1p5") or (dQ == "Q0.5EQ1.5"):
        return [[Q, L], (exprP, exprGs)]
    else: # the only case that we fit P directly 
        assert dQ == "QUAD4P", "Invalid choice for dQ" 
        return [[Q, L], (exprA, exprGs)]


def balance_kron_precond(Q):
    """
    In place balancing the dynamic ranges of the factors of Q to avoid over/under-flow.
    """
    order = len(Q)  # order of tensor or the number of factors in Q 
    if order>1:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = torch.prod(torch.stack(norms))**(1/order) # geometric mean 
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i]) 


def update_precond_kron_eq(QL, exprs, V, Hvp, lr=0.1, betaL=0.9):
    """
    The raw function for updating the Kron preconditioner Q and Lipschitz smoothness constant L with pair (V, Hvp),
    where Q is update as dQ = E*Q, 
    the pair (V, Hvp) can be (vector, hess-vector-prod) or (randn, gradient/momentum).  
    The damping logic is not included here. 
    """
    Q, L = QL
    _, exprGs, exprA = exprs
        
    def solve_triangular_right(B, A):
        # return B @ inv(A)
        if B.dim()>1: 
            return torch.linalg.solve_triangular(lift2single(A), lift2single(B), upper=True, left=False).to(B.dtype)
        else: # torch.linalg.solve_triangular complains if B.dim() < 2. So insert None.
            return (torch.linalg.solve_triangular(lift2single(A), lift2single(B[None,:]), upper=True, left=False)[0]).to(B.dtype)     
    
    A = exprA(*Q, Hvp)

    order = V.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1]) # permute dims like [0,1,2,3,4] -> [1,2,3,4,0]
    for i, q in enumerate(Q):
        conjB = conjB/q if q.dim()<2 else solve_triangular_right(conjB, q)
        if i < order - 1: # transpose dims like [1,2,3,4,0]->[0,2,3,4,1]->[0,1,3,4,2]->[0,1,2,4,3]->[0,1,2,3,4]
            conjB = torch.transpose(conjB, i, order - 1) 

    for i, q in enumerate(Q):
        term1 = exprGs[i](A, A.conj())
        term2 = exprGs[i](conjB.conj(), conjB)
                   
        if q.dim() < 2: # q is a diagonal matrix or scalar preconditioner
            ell = torch.max(torch.real(term1 + term2))
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 - term2) * q) # q.mul_(1 - lr/L[i] * (term1 - term2)): larger roundoff errors       
        else: # q is a matrix preconditioner 
            ell = norm_lower_bound_spd(term1 + term2)
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * torch.triu(term1 - term2) @ q)

    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def precond_grad_kron(QL, exprs, G):
    """
    Precondition gradient G with Kron preconditioner Q. 
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G) 


def update_precond_kron_whiten_eq(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = E*Q.
    """
    V = torch.randn_like(G)
    update_precond_kron_eq(QL, exprs, V, G + damping*V, lr=lr, betaL=betaL)
    

def update_precond_kron_whiten_qep(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q*E*P. 
    """   
    Q, L = QL
    exprP, exprGs, exprQs = exprs
    
    # balancing is not optional as L for each factor is not scaling invariant 
    balance_kron_precond(Q) 

    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G)) 
    for i, q in enumerate(Q):
        QPg = exprQs[i](q, Pg)
        term1 = exprGs[i](QPg, QPg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() * q * q.conj()
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] * q @ q.H
            ell = norm_lower_bound_spd(term1 + term2)
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 - term2) @ q)


def update_precond_kron_whiten_qeq(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q*E*Q. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    
    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G)) 
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (q @ term1 - q * term2))
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_whiten_q0p5eq1p5(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q as dQ = Q^0.5 * E * Q^1.5. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    
    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G)) 
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2  
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 @ q - term2 * q))
            procrustes_step(q)
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_whiten_quad(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron preconditioner Q with a quadratic form. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    
    total_numel = G.numel() 
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping*torch.randn_like(G))   
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/2/L[i] * (term1 - term2)
            q.mul_(gain * gain) 
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            p = q - lr/2/L[i] * (term1 @ q - term2 * q) 
            p = p - lr/2/L[i] * (p @ term1 - p * term2) 
            q.data = (p + p.H)/2 # p must be symmetric/hermitian  
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_whiten_quad4p(QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Almost the same as function update_precond_kron_whiten_quad except that fitting P directly. 
    This is the only case that we fit P directly (Q here is P). Vulnerable to numerical errors.  
    """   
    Q, L = QL
    exprA, exprGs = exprs

    total_numel = G.numel() 
    Pg = exprA(*Q, G + damping*torch.randn_like(G)) # Q actually is P; so just applying all its factors once.
    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            term2 = total_numel/q.numel() # times I
            ell = torch.max(torch.real(term1)) + term2 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/L[i] * (term1 - term2)
            q.mul_(gain * gain) 
        else: # matrix Q
            term2 = total_numel/q.shape[0] # times I
            ell = norm_lower_bound_spd(term1) + term2
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            p = q - lr/L[i] * (term1 @ q - term2 * q) 
            p = p - lr/L[i] * (p @ term1 - p * term2) 
            q.data = (p + p.H)/2 # p must be symmetric/hermitian  
            
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


class KronWhiten:
    """
    Implements the PSGD optimizer with the Kronecker product gradient/momentum whitening preconditioner. 
    Most of the time, the hyperparameter name says it all. Here are some comments on a few key hyperparameters.  
 
    1, preconditioner_max_size and preconditioner_max_skew. These two together control the complexity of the preconditioners. 
    For example, we are to precondition a 2D gradient with shape 10 x 50. 
    With preconditioner_max_size 20, we use a dense preconditioner for the first dim since 10 <= 20 and diagonal preconditioner for the second dim since 50 > 20. 
    With preconditioner_max_skew 1.5, we use a dense preconditioner for the first dim since 10/50 <= 1.5 and diagonal preconditioner for the second dim since 50/10 > 1.5.
 
    2, grad_clip_max_amp, betaL and damping. These three together help to stabilize the training. 
    PSGD here tries to normalize the gradients to unit amplitude. This can be problematic when gradients approach zeros. 
    The most effective way is to clip the preconditioned gradients if their amplitudes exceed grad_clip_max_amp, say 1.0.
    Another way is to damp and upper bound the fitted preconditioner such that P < eye/damping.   
    For extremely sparse gradients, increasing betaL (say to 0.999) helps a lot, where betaL is the EMA factor for the L-smoothness constant (wrt Q) estimation. 

    3, Lastly, dQ is for the selection of geometry for preconditioner update. QEQ, QUAD and Q0p5EQ1p5 all are good choices. 
    Q is initialized to preconditioner_init_scale * eye. Boolean setting whiten_grad decides to whiten whether the gradient or momentum. 
    Always good to check https://arxiv.org/abs/2402.11858 for math details. 
    """
    def __init__(self,  params_with_grad, 
                 preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale:float|None=None,
                 lr_params=0.001, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0,
                 grad_clip_max_amp=float("inf"), preconditioner_update_probability=1.0, whiten_grad=True, dQ="Q0.5EQ1.5"):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner 
        self.betaL = betaL # beta for the Lipschitz smoothness constant estimation; set to a large value for sparse gradients
        self.damping = damping # to damp and upper bound the preconditioner such that P < eye/damping  
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_amp = grad_clip_max_amp # clip grad once its average amplitude exceeds this max amplitude setting 
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        self._num_params = sum([p.numel() for p in self._params_with_grad])
        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly 
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p.squeeze(), preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew, dQ) for p in self._params_with_grad]
        self._ms, self._counter_m = None, 0 # momentum buffers and counter  
        self._whiten_grad = whiten_grad # set to False to whiten momentum.  
        if not whiten_grad:
            assert self.momentum > 0, "Cannot whiten momentum if the momentum setting is zero."
            print(f"Recommend to reduce lr_params by {int(((1 + momentum)/(1 - momentum))**0.5)} times")
        self._dQ = dQ
        if dQ == "QUAD4P": # the only case that we fit P directly 
            assert max([torch.finfo(p.dtype).eps for p in self._params_with_grad]) < 1e-6, "Directly fitting P needs at least single precision"
            self._update_precond = update_precond_kron_whiten_quad4p
            self._precond_grad = lambda QL, exprs, G: exprs[0](*QL[0], G) # it's exprA(*Q, G) 
        else:
            self._precond_grad = precond_grad_kron            
            if dQ == "QEP":
                self._update_precond = update_precond_kron_whiten_qep
            elif dQ == "EQ":
                self._update_precond = update_precond_kron_whiten_eq
            elif dQ == "QEQ":
                self._update_precond = update_precond_kron_whiten_qeq
            elif dQ == "QUAD":
                self._update_precond = update_precond_kron_whiten_quad
            else:
                assert (dQ == "Q0.5EQ1.5") or (dQ == "Q0p5EQ1p5"), "Invalid choice for dQ"
                self._update_precond = update_precond_kron_whiten_q0p5eq1p5


    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of PSGD with the Kronecker product gradient/momentum whitening preconditioner.
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = [g.squeeze() for g in torch.autograd.grad(loss, self._params_with_grad)]
            
        if self._QLs_exprs is None:
            scale = max([torch.mean((torch.abs(g))**4) for g in grads])
            scale = (scale + self.damping**4)**(-1/8)
            self._QLs_exprs = [init_kron(g, scale, self._preconditioner_max_size, self._preconditioner_max_skew, self._dQ) for g in grads]
            
        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]

            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
        else:
            self._ms, self._counter_m = None, 0

        if torch.rand([]) < self.preconditioner_update_probability: # update Q
            if self._whiten_grad: # Q whitens gradient 
                [self._update_precond(*QL_exprs, g, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping) 
                 for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            else: # Q whitens momentum 
                [self._update_precond(*QL_exprs, m, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping) 
                 for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
                
        if self.momentum > 0: # precondition momentum 
            pre_grads = [self._precond_grad(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else: # precondition gradient 
            pre_grads = [self._precond_grad(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]

        lr = self.lr_params
        if self.grad_clip_max_amp < float("inf"): # clip preconditioned gradient 
            avg_amp = torch.sqrt(torch.real(sum([torch.sum(g*g.conj()) for g in pre_grads]))/self._num_params)
            if avg_amp > self.grad_clip_max_amp:
                lr = lr * self.grad_clip_max_amp / avg_amp
            
        # Update the parameters. 
        [param.subtract_(lr*g.view_as(param)) for (param, g) in zip(self._params_with_grad, pre_grads)]        

        # return whatever closure returns
        return closure_returns
    

def update_precond_kron_newton_eq(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron Newton-type preconditioner Q as dQ = E*Q with a pair of vector and hvp, (V, Hvp). 
    """    
    update_precond_kron_eq(QL, exprs, V, Hvp + damping*torch.randn_like(Hvp), lr=lr, betaL=betaL)


def update_precond_kron_newton_qep(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron Newton-type preconditioner Q as dQ = Q*E*P with a pair of vector and hvp, (V, Hvp). 
    """   
    Q, L = QL
    exprP, exprGs, exprQs = exprs

    # balancing is not optional as L for each factor is not scaling invariant 
    balance_kron_precond(Q) 
    Ph = exprP(*[q.conj() for q in Q], *Q, Hvp + damping*torch.randn_like(Hvp)) 

    for i, q in enumerate(Q):
        QPh = exprQs[i](q, Ph)
        Qv = exprQs[i](q, V)
        term1 = exprGs[i](QPh, QPh.conj())
        term2 = exprGs[i](Qv, Qv.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            ell = norm_lower_bound_spd(term1 + term2) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 - term2) @ q)


def update_precond_kron_newton_qeq(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron Newton-type preconditioner Q as dQ = Q*E*Q with a pair of vector and hvp, (V, Hvp). 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    Ph = exprP(*[q.conj() for q in Q], *Q, Hvp + damping*torch.randn_like(Hvp)) 

    for i, q in enumerate(Q):
        term1 = exprGs[i](Ph, Ph.conj())
        term2 = exprGs[i](V, V.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            ell = norm_lower_bound_spd(term1 + term2) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * q @ (term1 - term2))
    
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_newton_q0p5eq1p5(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron Newton-type preconditioner Q as dQ = Q^0.5 * E * Q^1.5 with a pair of vector and hvp, (V, Hvp). 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    Ph = exprP(*[q.conj() for q in Q], *Q, Hvp + damping*torch.randn_like(Hvp)) 

    for i, q in enumerate(Q):
        term1 = exprGs[i](Ph, Ph.conj())
        term2 = exprGs[i](V, V.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.mul_(1 - lr/L[i] * (term1 - term2))
        else: # matrix Q
            ell = norm_lower_bound_spd(term1 + term2) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            q.sub_(lr/L[i] * (term1 - term2) @ q)
            procrustes_step(q)
    
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_newton_quad(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the Kron Newton-type preconditioner Q with a quadratic form for dQ and pair of vector and hvp, (V, Hvp). 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    Ph = exprP(*[q.conj() for q in Q], *Q, Hvp + damping*torch.randn_like(Hvp)) 

    for i, q in enumerate(Q):
        term1 = exprGs[i](Ph, Ph.conj())
        term2 = exprGs[i](V, V.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/2/L[i] * (term1 - term2)
            q.mul_(gain * gain)
        else: # matrix Q
            ell = norm_lower_bound_spd(term1 + term2) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            err = lr/2/L[i] * (term1 - term2)
            p = q - err @ q     # p = q - lr/L[i]/2 * (term1 - term2) @ q
            p = p - p @ err     # p = p - lr/L[i]/2 * p @ (term1 - term2)
            q.data = (p + p.H)/2 # p must be symmetric or hermitian  
    
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


def update_precond_kron_newton_quad4p(QL, exprs, V, Hvp, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Almost the same as function update_precond_kron_newton_quad except that we fit P directly. 
    This is the only case that fits P directly (Q here is P). It's vulnerable to numerical errors.   
    """   
    Q, L = QL
    exprA, exprGs = exprs
    Ph = exprA(*Q, Hvp + damping*torch.randn_like(Hvp)) # Q actually is P; so only need to apply its factors once.  

    for i, q in enumerate(Q):
        term1 = exprGs[i](Ph, Ph.conj())
        term2 = exprGs[i](V, V.conj())
        if q.dim() < 2: # diagonal or scalar Q 
            ell = torch.max(torch.real(term1 + term2)) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            gain = 1 - lr/L[i] * (term1 - term2)
            q.mul_(gain * gain)
        else: # matrix Q
            ell = norm_lower_bound_spd(term1 + term2) 
            L[i].data = torch.max(betaL*L[i] + (1 - betaL)*ell, ell)
            err = lr/L[i] * (term1 - term2)
            p = q - err @ q     # p = q - lr/L[i] * (term1 - term2) @ q
            p = p - p @ err     # p = p - lr/L[i] * p @ (term1 - term2)
            q.data = (p + p.H)/2 # p must be symmetric or hermitian  
    
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)


class KronNewton:
    """
    Implements the Kronecker product Newton-type preconditioner as a class. 
    Most of the time, the hyperparameter name says it all. Here are some comments on a few key parameters.  

    1, preconditioner_max_size and preconditioner_max_skew. These two together control the complexity of the preconditioners. 
    For example, we are to precondition a 2D gradient with shape 10 x 50. 
    With preconditioner_max_size 20, we use a dense preconditioner for the first dim since 10 <= 20 and diagonal preconditioner for the second dim since 50 > 20. 
    With preconditioner_max_skew 1.5, we use a dense preconditioner for the first dim since 10/50 <= 1.5 and diagonal preconditioner for the second dim since 50/10 > 1.5.

    2, grad_clip_max_norm, betaL and damping. These three together help to stabilize the training. 
    The grad_clip_max_norm is used to clip the preconditioned gradient to stabilize the optimization as in the classic trust region method. 
    Setting damping is used to damp and upper bound the fitted preconditioner such that P < eye/damping. 
    For extremely sparse Hess-vector-prod, a large betaL (say 0.999) helps a lot, where betaL is the EMA factor for the L-smoothness constant (wrt Q) estimation. 

    3, exact_hessian_vector_product. 
    By setting this flag to False, the finite difference method will be used for Hvp approximation. 
    Be cautious with the finite difference method (possible numerical issues; the closure must behave like a pure function).

    4, Lastly, dQ is for the selection of geometry for preconditioner update. QEQ, QUAD and Q0p5EQ1p5 all are good choices. 
    Both lr_params and lr_preconditioner are normalized learning rates. 
    Q is initialized to preconditioner_init_scale * eye. 
    Always good to check https://arxiv.org/abs/2402.11858 for math details. 
    """
    def __init__(self,  params_with_grad, preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale:float|None=None,
                        lr_params=0.01, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0,
                        grad_clip_max_norm=float("inf"), preconditioner_update_probability=1.0,
                        exact_hessian_vector_product=True, dQ="Q0.5EQ1.5"):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner     
        self.betaL = betaL # beta for Lipschitz smoothness constant estimation; set to a large value for sparse Hvp  
        self.damping = damping # used to damp and upper bound P as P < eye/damping 
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        eps = max([torch.finfo(p.dtype).eps for p in self._params_with_grad])
        self._delta_param_scale = eps ** 0.5
        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly 
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p.squeeze(), preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew, dQ) for p in self._params_with_grad]
        self._ms, self._counter_m = None, 0 # momentum buffers and counter 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")
        self._dQ = dQ
        if dQ == "QUAD4P": # the only case that fits P directly 
            self._update_precond = update_precond_kron_newton_quad4p
            self._precond_grad = lambda QL, exprs, G: exprs[0](*QL[0], G) # it's exprA(*Q, G) 
            assert eps < 1e-6, "Directly fitting P needs at least single precision" 
        else:
            self._precond_grad = precond_grad_kron
            if dQ == "QUAD":
                self._update_precond = update_precond_kron_newton_quad
            elif dQ == "QEP":
                self._update_precond = update_precond_kron_newton_qep
            elif dQ == "EQ":
                self._update_precond = update_precond_kron_newton_eq
            elif dQ == "QEQ":
                self._update_precond = update_precond_kron_newton_qeq
            else:
                assert (dQ == "Q0.5EQ1.5") or (dQ == "Q0p5EQ1p5"), "Invalid choice for dQ"
                self._update_precond = update_precond_kron_newton_q0p5eq1p5            


    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of PSGD with the Kronecker product Newton-type preconditioner.  
        """
        if (torch.rand([]) < self.preconditioner_update_probability) or (self._QLs_exprs is None):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self._exact_hessian_vector_product:
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(p) for p in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs) # this line also works for complex matrices 
            else: # approximate the Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                    
                vs = [torch.randn_like(p) for p in self._params_with_grad]
                [p.add_(v, alpha=self._delta_param_scale) for (p, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [(perturbed_g - g)/self._delta_param_scale for (perturbed_g, g) in zip(perturbed_grads, grads)]               
                [p.sub_(v, alpha=self._delta_param_scale) for (p, v) in zip(self._params_with_grad, vs)] # remove the perturbation            
            
            if self._QLs_exprs is None: # initialize QLs on the fly if it is None 
                scale = (sum([torch.sum(torch.abs(v)**2) for v in vs])/sum([v.numel() for v in vs])) ** (1/4) # (mean(|v|^2))^(1/4)
                scale = scale * (max([torch.mean((torch.abs(h))**4) for h in Hvs]) + self.damping**4) ** (-1/8) # (mean(|v|^2))^(1/4) * (mean(|h|^4))^(-1/8)
                self._QLs_exprs = [init_kron(h.squeeze(), scale, self._preconditioner_max_size, self._preconditioner_max_skew, self._dQ) for h in Hvs]
            # update preconditioner
            [self._update_precond(*QL_exprs, v.squeeze(), h.squeeze(), lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping) 
             for (QL_exprs, v, h) in zip(self._QLs_exprs, vs, Hvs)]
        else: # only evaluate the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)

        grads = [g.squeeze() for g in grads]
        if self.momentum > 0: # precondition the momentum 
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                
            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
            pre_grads = [self._precond_grad(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else: # precondition the gradient 
            self._ms, self._counter_m = None, 0 # clear the buffer and counter when momentum is set to zero 
            pre_grads = [self._precond_grad(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            
        lr = self.lr_params
        if self.grad_clip_max_norm < float("inf"):
            grad_norm = torch.sqrt(torch.real(sum([torch.sum(g*g.conj()) for g in pre_grads])))
            if grad_norm > self.grad_clip_max_norm:
                lr = lr * self.grad_clip_max_norm / grad_norm
            
        # Update the parameters. 
        [param.subtract_(lr*g.view_as(param)) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        # return whatever closure returns
        return closure_returns


#############       End of PSGD Kronecker product preconditioners       #############


#############       Begin of PSGD LRA (low rank approximation) preconditioners       #############


def IpUVtmatvec(U, V, x):
    """
    Returns (I + U*V')*x. All variables are either matrices or column vectors. 
    """
    return x + U.mm(V.t().mm(x))


def update_precond_lra(UVd, Luvd, v, h, lr=0.1, betaL=0.9):
    """
    The raw function for updating the LRA preconditioner Q = (I + U*V')*diag(d) with pair (v, h), 
    where h can a Hvp associated with v, or a gradient/momentum independent of v.
    State variables (U, V, d) and their Lipschitz smoothness constant estimates (Lu, Lv, Ld) are updated inplace. 
    Damping logic is not implemented here.                  
    Note that U, V, d, v, and h all are either matrices or column vectors.  
    """
    U, V, d = UVd
    Lu, Lv, Ld = Luvd

    # Approximately balancing U and V such that U^T U = V^T V (exact balancing needs three EVDs)
    UtU, VtV = U.t() @ U, V.t() @ V
    trUtU, trVtV = torch.sum(UtU.diagonal()), torch.sum(VtV.diagonal())
    rho = (trUtU/trVtV) ** (1/4) # will scale U and V as U <-- U/rho and V <-- V*rho
    rho2 = rho * rho
    E = 0.1 * (UtU/rho2 - VtV*rho2)/(trUtU/rho2 + trVtV*rho2) # errors after scaling U and V  
    E2 = 0.5 * E @ E # using this E2 term to make (I - E + E^2/2)(I + E + E^2/2) = (I + E^2/2)^2 - E^2 = I + E^4/4 
    U.div_(rho), V.mul_(rho) # scale U and V to have ||U||_F = ||V||_F
    U.sub_(U @ (E - E2)), V.add_(V @ (E + E2)) # rotate (as tr(E)=0) U and V to approach U^TU = V^TV

    Qh = IpUVtmatvec(U, V, d * h)
    Ph = d*IpUVtmatvec(V, U, Qh)

    IpVtU = V.t().mm(U)
    IpVtU.diagonal().add_(1) # avoid forming matrix I explicitly 
    invQtv = v/d
    LU, pivots = torch.linalg.lu_factor(lift2single(IpVtU))
    invQtv = invQtv - V.mm(torch.linalg.lu_solve(LU, pivots, lift2single(U.t().mm(invQtv)), adjoint=True).to(V.dtype))
    invPv  = invQtv - U.mm(torch.linalg.lu_solve(LU, pivots, lift2single(V.t().mm(invQtv))).to(U.dtype))
    invPv = invPv/d

    # update d 
    Phh, vinvPv = Ph*h, v*invPv
    ell = torch.max(torch.abs(Phh)) + torch.max(torch.abs(vinvPv))
    Ld.data = torch.max(betaL*Ld + (1 - betaL)*ell, ell)
    d.sub_(lr/Ld*(Phh - vinvPv)*d)  # d.mul_(1 - lr/Ld*(Phh - vinvPv)): larger roundoff errors, unstable with bfloat16 and lr<<1 

    a, b = Qh, invQtv        
    if torch.rand([]) < 0.5: # only update U
        atV = a.t().mm(V)
        btV = b.t().mm(V)
        atVVt = atV.mm(V.t())
        btVVt = btV.mm(V.t())
        ell = (torch.linalg.vector_norm(a)*torch.linalg.vector_norm(atVVt) + 
               torch.linalg.vector_norm(b)*torch.linalg.vector_norm(btVVt))
        Lu.data = torch.max(betaL*Lu + (1 - betaL)*ell, ell)
        U.sub_(lr/Lu * ( a.mm(atV.mm(IpVtU)) - b.mm(btV.mm(IpVtU)) ))
    else: # only udate V
        atU = a.t().mm(U)
        btU = b.t().mm(U)
        UUta = U.mm(atU.t())
        UUtb = U.mm(btU.t())
        ell = (torch.linalg.vector_norm(a)*torch.linalg.vector_norm(UUta) + 
               torch.linalg.vector_norm(b)*torch.linalg.vector_norm(UUtb))
        Lv.data = torch.max(betaL*Lv + (1 - betaL)*ell, ell)
        V.sub_(lr/Lv * ( (a + V.mm(atU.t())).mm(atU) - (b + V.mm(btU.t())).mm(btU) ))


def precond_grad_lra(UVd, g):
    """
    Precondition gradient g with Q = (I + U*V')*diag(d).                                      
    All variables here are either matrices or column vectors. 
    """
    U, V, d = UVd
    g = IpUVtmatvec(U, V, d * g)
    g = d * IpUVtmatvec(V, U, g)
    return g


def update_precond_lra_whiten(UVd, Luvd, g, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the LRA whiten preconditioner. 
    """
    v = torch.randn_like(g)
    update_precond_lra(UVd, Luvd, v, g + damping*v, lr=lr, betaL=betaL)


class LRAWhiten:
    """
    Implements the PSGD LRA gradient/momentum whitening preconditioner as a class. 
    Most of the time, the hyperparameter name says it all. Here are some comments on a few key parameters.  

    1, rank_of_approximation. 
    Preconditioner Q has a diagonal part and a low rank part, whose rank is decided by this setting. 
    Rank 0 reduces Q to a diagonal preconditioner. 
 
    2, grad_clip_max_amp, betaL and damping. These three together help to stabilize the training. 
    PSGD here tries to normalize the gradients to unit amplitude. This can be problematic when gradients approach zeros. 
    The most effective way is to clip the preconditioned gradients when their amplitudes exceed grad_clip_max_amp, say 1.0. 
    Another way is to damp and upper bound the fitted preconditioner as P < eye/damping. 
    For extremely sparse gradient, increasing betaL (say to 0.999) also helps a lot, where betaL is the EMA factor for the L-smoothness constant (wrt Q) estimation. 
    
    3, Lastly, Q is initialized to preconditioner_init_scale * eye. 
    Boolean setting whiten_grad decides to whiten whether the gradient or momentum. 
    Always good to check https://arxiv.org/abs/2402.11858 for math details. 
    """
    def __init__(self,  params_with_grad, rank_of_approximation:int=10, preconditioner_init_scale:float|None=None,
                        lr_params=0.001, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0,
                        grad_clip_max_amp=float("inf"), preconditioner_update_probability=1.0, whiten_grad=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.betaL = betaL  # set to a large betaL for sparse gradients 
        self.damping = damping # to damp and upper bound P as P < eye/damping
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_amp = grad_clip_max_amp
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        assert 0 <= rank_of_approximation < num_params, "Rank r should be in range [0, number of total parameters)"
        self._UVd = [] # saves U, V and d
        self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device)) # U
        self._UVd[0] *= 0.1**0.5 / torch.linalg.vector_norm(self._UVd[0])
        self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device)) # V
        self._UVd[1] *= 0.1**0.5 / torch.linalg.vector_norm(self._UVd[1])
        if preconditioner_init_scale is None:
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._UVd.append(torch.ones(num_params, 1, dtype=dtype, device=device) * preconditioner_init_scale)
        self._Luvd = [lift2single(torch.zeros([], dtype=dtype, device=device)) for _ in range(3)]
        self._m, self._counter_m = None, 0 # momentum buffer and counter 
        self._whiten_grad = whiten_grad
        if (not whiten_grad):
            assert self.momentum > 0, "Cannot whiten momentum if the momentum setting is zero."
            print(f"Recommend to reduce lr_params by {int(((1 + momentum)/(1 - momentum))**0.5)} times")


    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of the PSGD LRA gradient/momentum whitening optimizer. 
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)

        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) # column vector 
        
        if len(self._UVd) < 3: # initialize d on the fly 
            self._UVd.append((torch.mean(grad**4) + self.damping**4)**(-1/8) * torch.ones_like(grad)) 

        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._m is None:
                self._m = torch.zeros_like(grad)

            self._m.mul_(beta).add_(grad, alpha=1 - beta) 
        else: # clear the momentum buffer and counter when momentum is set to zero
            self._m, self._counter_m = None, 0 

        if torch.rand([]) < self.preconditioner_update_probability: # update preconditioner 
            if self._whiten_grad: # whitens gradient 
                update_precond_lra_whiten(self._UVd, self._Luvd, grad, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)
            else: # whitens momentum 
                update_precond_lra_whiten(self._UVd, self._Luvd, self._m, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)

        if self.momentum > 0: # precondition momentum 
            pre_grad = precond_grad_lra(self._UVd, self._m)
        else: # precondition gradient 
            pre_grad = precond_grad_lra(self._UVd, grad)
            
        lr = self.lr_params
        if self.grad_clip_max_amp < float("inf"): # clip preconditioned gradient 
            amp = torch.sqrt(torch.mean(pre_grad * pre_grad))
            if amp > self.grad_clip_max_amp:
                lr = lr * self.grad_clip_max_amp/amp 
            
        # update the parameters 
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

def update_precond_lra_newton(UVd, Luvd, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update the LRA Newton preconditioner. 
    """
    update_precond_lra(UVd, Luvd, v, h + damping*torch.randn_like(h), lr=lr, betaL=betaL)


class LRANewton:
    """
    Implements the PSGD LRA Newton-type preconditioner as a class. 
    Most of the time, the hyperparameter name says it all. Here are some comments on a few key parameters.  

    1, rank_of_approximation. 
    Preconditioner Q has a diagonal part and a low rank part, whose rank is decided by this setting. 
    Rank 0 reduces Q to a diagonal preconditioner. 

    2, grad_clip_max_norm, betaL and damping. These three together help to stabilize the training.
    The grad_clip_max_norm is used to clip the preconditioned gradient to stabilize the optimization as in the classic trust region method.
    Setting damping is used to damp and upper bound the preconditioner as P < eye/damping. 
    For extremely sparse hess-vector-prods, a large betaL (say 0.999) helps a lot, where betaL is the EMA factor for the L-smoothness constant (wrt Q) estimation.

    3, exact_hessian_vector_product. 
    By setting this flag to False, the finite difference method will be used for Hvp approximation. 
    Be cautious with the finite difference method (possible numerical issues; the closure must behave like a pure function).

    4, Lastly, Q is initialized to preconditioner_init_scale * eye. 
    Both lr_params and lr_preconditioner are normalized learning rates. 
    Always good to check https://arxiv.org/abs/2402.11858 for math details.
    """
    def __init__(self,  params_with_grad, rank_of_approximation:int=10, preconditioner_init_scale:float|None=None,
                        lr_params=0.01, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0,
                        grad_clip_max_norm=float("inf"), preconditioner_update_probability=1.0,
                        exact_hessian_vector_product=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.betaL = betaL # set to a large betaL for sparse Hvp 
        self.damping = damping # to damp and upper bound the preconditioner as P < eye/damping 
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._delta_param_scale = torch.finfo(dtype).eps**0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        assert 0 <= rank_of_approximation < num_params, "Rank r should be in range [0, number of total parameters)"
        self._UVd = [] # saves U, V and d
        self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device)) # U
        self._UVd[0] *= 0.1**0.5 / torch.linalg.vector_norm(self._UVd[0])
        self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device)) # V
        self._UVd[1] *= 0.1**0.5 / torch.linalg.vector_norm(self._UVd[1])
        if preconditioner_init_scale is None:
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._UVd.append(torch.ones(num_params, 1, dtype=dtype, device=device) * preconditioner_init_scale)
        self._Luvd = [lift2single(torch.zeros([], dtype=dtype, device=device)) for _ in range(3)]
        self._m, self._counter_m = None, 0 # momentum buffer and counter 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")


    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of the PSGD LRA Newton optimizer. 
        """
        if (torch.rand([]) < self.preconditioner_update_probability) or (len(self._UVd) < 3):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self._exact_hessian_vector_product:
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(param) for param in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs)
            else: # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                
                vs = [torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v, alpha=self._delta_param_scale) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [(perturbed_g - g)/self._delta_param_scale for (perturbed_g, g) in zip(perturbed_grads, grads)]
                [param.sub_(v, alpha=self._delta_param_scale) for (param, v) in zip(self._params_with_grad, vs)]

            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) # column vector
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) # column vector  
            if len(self._UVd) < 3: # init d if it's not in the UVd list 
                self._UVd.append((torch.mean(v*v))**(1/4) * (torch.mean(h**4) + self.damping**4)**(-1/8) * torch.ones_like(v))
            
            # update preconditioner
            update_precond_lra_newton(self._UVd, self._Luvd, v, h, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)
        else: # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            
        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) # column vector 

        if self.momentum > 0: # precondition momentum  
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._m is None:
                self._m = torch.zeros_like(grad)

            self._m.mul_(beta).add_(grad, alpha=1 - beta)
            pre_grad = precond_grad_lra(self._UVd, self._m)
        else: # precondition gradient 
            self._m, self._counter_m = None, 0 # clear the buffer and counter when momentum is set to zero 
            pre_grad = precond_grad_lra(self._UVd, grad)
            
        lr = self.lr_params
        if self.grad_clip_max_norm < float("inf"):
            grad_norm = torch.linalg.vector_norm(pre_grad)
            if grad_norm > self.grad_clip_max_norm:
                lr = lr * self.grad_clip_max_norm / grad_norm
            
        # update the parameters
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

#############       End of PSGD LRA preconditioners       #############


#############       Begin of PSGD dense matrix Newton-type preconditioner       #############


def update_precond_dense_eq(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update dense matrix Newton-type preconditioner Q with local coordinate dQ = mathcal{E} * Q.
    """
    a = Q.mm(h + damping*torch.randn_like(h))
    b = torch.linalg.solve_triangular(lift2single(Q.t()), lift2single(v), upper=False).to(v.dtype)
    ell = torch.sum(a*a + b*b)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    Q.sub_(lr/L * torch.triu(a.mm(a.t()) - b.mm(b.t())) @ Q)


def update_precond_dense_qep(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update dense matrix Newton-type preconditioner Q with local coordinate dQ = Q * mathcal{E} * P.
    """
    a = Q @ (Q.T @ (Q @ (h + damping*torch.randn_like(h))))
    b = Q @ v
    ell = torch.sum(a*a + b*b)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    Q.sub_(lr/L * (a @ (a.T @ Q) - b @ (b.T @ Q)))


def update_precond_dense_qeq(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update dense matrix Newton-type preconditioner Q with local coordinate dQ = Q * mathcal{E} * Q.
    """
    a = Q.T @ (Q @ (h + damping*torch.randn_like(h)))
    ell = torch.sum(a*a + v*v)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    Q.sub_(lr/L * ((Q @ a) @ a.T - (Q @ v) @ v.T))


def update_precond_dense_q0p5eq1p5(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update dense matrix Newton-type preconditioner Q with local coordinate dQ = Q^0.5 * mathcal{E} * Q^1.5.
    """
    a = Q.T @ (Q @ (h + damping*torch.randn_like(h)))
    ell = torch.sum(a*a + v*v)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    Q.sub_(lr/L * (a @ (a.T @ Q) - v @ (v.T @ Q)))
    procrustes_step(Q)


def update_precond_dense_quad(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    Update dense matrix Newton-type preconditioner Q with a quadratic form for dQ.
    """
    a = Q @ (Q @ (h + damping*torch.randn_like(h))) # Q is symmetric here 
    ell = torch.sum(a*a + v*v)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    p = Q - lr/2/L * (a @ (a.T @ Q) - v @ (v.T @ Q)) 
    p = p - lr/2/L * ((p @ a) @ a.T - (p @ v) @ v.T) 
    Q.data = (p + p.T)/2 


def update_precond_dense_quad4p(Q, L, v, h, lr=0.1, betaL=0.9, damping=1e-9):
    """
    The only case that fits P directly. 
    """
    a = Q @ (h + damping*torch.randn_like(h)) # Q actually is P; so just apply it once. 
    ell = torch.sum(a*a + v*v)
    L.data = torch.max(betaL*L + (1 - betaL)*ell, ell)
    p = Q - lr/L * (a @ (a.T @ Q) - v @ (v.T @ Q)) 
    p = p - lr/L * ((p @ a) @ a.T - (p @ v) @ v.T) 
    Q.data = (p + p.T)/2 


class DenseNewton:
    """
    Implements the PSGD dense matrix Newton-type preconditioner as a class. 
    Be extra cautious when using the finite difference method for Hvp approximation (the closure must behave like a pure function).
    It's mainly for illustrating how PSGD works due to its simplicity. 
    It's also a good alternative to the BFGS like quasi-Newton methods as no line search is required. 
    """
    def __init__(self, params_with_grad, preconditioner_init_scale:float|None=None,
                 lr_params=0.01, lr_preconditioner=0.1, betaL=0.9, damping=1e-9, momentum=0.0, 
                 grad_clip_max_norm=float("inf"), preconditioner_update_probability=1.0,
                 exact_hessian_vector_product=True, dQ="Q0.5EQ1.5"):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.betaL = betaL  # set to a large betaL for sparse Hvp  
        self.damping = damping # to damp and upper bound the preconditioner as P < eye/damping
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]  # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._delta_param_scale = torch.finfo(dtype).eps ** 0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if preconditioner_init_scale is None: # initialize Q on the fly
            self._Q = None 
        else:
            if dQ == "QUAD4P": # Q actually is P 
                preconditioner_init_scale *= preconditioner_init_scale
            self._Q = torch.eye(num_params, dtype=dtype, device=device) * preconditioner_init_scale
        self._L = lift2single(torch.zeros([], dtype=dtype, device=device)) # Lipschitz smoothness constant estimation for the psgd criterion 
        self._m, self._counter_m = None, 0 # buffer and counter for momentum 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")
        self._dQ = dQ
        if dQ == "QUAD4P": # the only case that we fit P directly
            self._update_precond = update_precond_dense_quad4p
            self._precond_grad = lambda Q, g: Q @ g
            assert torch.finfo(dtype).eps < 1e-6, "Directly fitting P needs at least single precision" 
        elif dQ == "QUAD":
            self._update_precond = update_precond_dense_quad
            self._precond_grad = lambda Q, g: Q @ (Q @ g) # Q is symmetric; just save one transpose 
        else:
            self._precond_grad = lambda Q, g: Q.T @ (Q @ g)
            if dQ == "QEP":
                self._update_precond = update_precond_dense_qep
            elif dQ == "EQ":
                self._update_precond = update_precond_dense_eq  
            elif dQ == "QEQ":
                self._update_precond = update_precond_dense_qeq
            else: 
                assert (dQ == "Q0p5EQ1p5") or (dQ == "Q0.5EQ1.5"), "Invalid choice for dQ"
                self._update_precond = update_precond_dense_q0p5eq1p5
                        

    @torch.no_grad()
    def step(self, closure):
        """
        Performs one step of PSGD with the dense matrix Newton-type preconditioner. 
        """
        if (torch.rand([]) < self.preconditioner_update_probability) or (self._Q is None):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self._exact_hessian_vector_product: # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(param) for param in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs)
            else: # approximate Hessian-vector product via finite-difference formulae. Use it with cautions.
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad)
                
                vs = [torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v, alpha=self._delta_param_scale) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [(perturbed_g - g)/self._delta_param_scale for (perturbed_g, g) in zip(perturbed_grads, grads)]
                [param.sub_(v, alpha=self._delta_param_scale) for (param, v) in zip(self._params_with_grad, vs)]

            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) 
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) 
            if self._Q is None: # initialize Q on the fly if it is None
                scale = (torch.mean(v*v))**(1/4) * (torch.mean(h**4) + self.damping**4)**(-1/8)
                if self._dQ == "QUAD4P": # Q actually is P in this case 
                    scale *= scale 
                self._Q = torch.eye(len(v), dtype=v.dtype, device=v.device) * scale

            # update preconditioner 
            self._update_precond(self._Q, self._L, v, h, lr=self.lr_preconditioner, betaL=self.betaL, damping=self.damping)
        else: # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)
            
        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) 
           
        if self.momentum > 0: # precondition momentum 
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._m is None:
                self._m = torch.zeros_like(grad)

            self._m.mul_(beta).add_(grad, alpha=1 - beta)
            pre_grad = self._precond_grad(self._Q, self._m)
        else:
            self._m, self._counter_m = None, 0 # clear the buffer and counter when momentum is set to zero 
            pre_grad = self._precond_grad(self._Q, grad)
        
        lr = self.lr_params
        if self.grad_clip_max_norm < float("inf"):
            grad_norm = torch.linalg.vector_norm(pre_grad)
            if grad_norm > self.grad_clip_max_norm:
                lr = lr * self.grad_clip_max_norm / grad_norm

        # update the parameters
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

#############       End of PSGD dense matrix Newton-type preconditioner       #############

""" end of psgd """
