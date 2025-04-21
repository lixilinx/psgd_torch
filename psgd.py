"""
The new tri-solver-free PSGD-Kron preconditioner is derived with local coordinate dQ = Q * mathcal{E} * P. 
The PSGD-LRA preconditioner still uses local coordinate dQ = mathcal{E} * Q (Lie group), and needs to a small linear solver.
I also keep the PSGD dense matrix Newton-type preconditioner here to illustrate the math. 
It's also a good alternative of BFGS like quasi-Newton optimizers as no line search is required. 

Xi-Lin Li, lixilinx@gmail.com, April, 2025. 
Main refs: https://arxiv.org/abs/1512.04202; https://arxiv.org/abs/2402.11858. 
"""

import opt_einsum
import torch


def norm_lower_bound_herm(A):
    """
    Returns a cheap lower bound for the spectral norm of a symmetric or Hermitian matrix A.
    """
    max_abs = torch.max(torch.abs(A)) # used to normalize A to avoid numerically under/over-flow
    if max_abs > 0:
        A = A/max_abs
        aa = torch.real(A * A.conj())
        _, j = torch.max(torch.sum(aa, dim=1), 0)
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))
    else: # must have A=0
        return max_abs 
    
    
#############       Begin of PSGD Kronecker product preconditioners       #############         


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0):
    """
    For a scalar or tensor t, we initialize its states (preconditioner Q and Lipschitz constant L), 
    and reusable contraction expressions for updating Q and preconditioning gradient.
    
    1, The preconditioner Q is initialized to 
        Q = Scale * I = Scale * kron(eye(t.shape[0]), eye(t.shape[1]), ...)
       where the eye(.) may be replaced with diag(ones(.)) if that dim is too large, determined by max_size and max_skew.
       
       The Lipschitz constant L is initialized to zero. 
       
    2, A series of enisum contract expressions. The following subscript examples are for a 5th order tensor.  
        2.1, exprP is the expression for calculating the preconditioned gradient, e.g.,
                'aA,bB,cC,dD,eE,aα,bβ,cγ,dδ,eε,αβγδε->ABCDE'
        2.2, exprGs is a list of expressions for calculating the gradients wrt Q on each dim, e.g.,
                'abCde,abγde->Cγ'
            for the middle dim of a 5th order tensor Q . 
    """
    shape = t.shape 
    if len(shape)==0: # scalar 
        Q = [Scale * torch.ones_like(t),]
        L = [torch.zeros_like(t.real),]
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape) 
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape),]
    else: # tensor 
        if len(shape) > 26:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters; Replace 26 with larger numbers!")   
            
        scale = Scale ** (1/len(shape)) 
    
        Q, L = [], []
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = [], [], "", "" # used for getting the subscripts for exprP
        for i, size in enumerate(shape):
            L.append(torch.zeros([], dtype=t.real.dtype, device=t.device))
            if size == 1 or size > max_size or size**2 > max_skew * t.numel():
                # use diagonal matrix as preconditioner for this dim 
                Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
                
                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i+26)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))
            else: # use matrix preconditioner for this dim 
                Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))
                
                a, b, c = opt_einsum.get_symbol(i), opt_einsum.get_symbol(i + 26), opt_einsum.get_symbol(i + 805)
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b
                
                piece1 = "".join([opt_einsum.get_symbol(i+26) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                piece2 = "".join([opt_einsum.get_symbol(i+805) if j==i else opt_einsum.get_symbol(j) for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + opt_einsum.get_symbol(i+26) + opt_einsum.get_symbol(i+805)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))
        
        subscripts = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        exprP = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape)
    
    exprGs = tuple(exprGs)
    return [[Q, L], (exprP, exprGs)]


def balance_kron_precond(Q):
    """
    Balance the dynamic ranges of the factors of Q to avoid over/under-flow.
    """
    order = len(Q)  # order of tensor or the number of factors in Q 
    if order>1:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = (torch.cumprod(torch.stack(norms), dim=0)[-1])**(1/order) # geometric mean 
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i]) 


def precond_grad_kron_whiten(QL, exprs, G, lr=0.01, updateP=True):
    """
    Precondition gradient G with Kron product gradient/momentum whitening preconditioner Q. 
    We just optionally update the preconditioner here to save computations. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    G = exprP(*[q.conj() for q in Q], *Q, G) 
    
    if updateP: # update preconditioner
        total_numel = G.numel() 
        for i, q in enumerate(Q):
            GGc = exprGs[i](G, G.conj())
            if q.dim() < 2: # diagonal Q 
                qGGcqh = q * GGc * q.conj()
                qqh = total_numel/q.numel() * q * q.conj()
                ell = torch.max(torch.abs(qGGcqh + qqh)) 
                L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
                q.sub_(lr/L[i] * (qGGcqh - qqh) * q)
            else: # matrix Q
                qGGcqh = q @ GGc @ q.H
                qqh = total_numel/q.shape[0] * q @ q.H
                ell = norm_lower_bound_herm(qGGcqh + qqh)
                L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
                q.sub_(lr/L[i] * (qGGcqh - qqh) @ q)
                
        if torch.rand([]) < 0.01: # balance factors of Q
            balance_kron_precond(Q)
                    
    return G
                    

class KronWhiten:
    """
    Implements the PSGD optimizer with Kronecker product gradient/momentum whitening preconditioner.
    By default, we whiten the gradient, not the momentum. 
    """
    def __init__(self,  params_with_grad, 
                 preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale:float|None=None,
                 lr_params=0.001, lr_preconditioner=0.01, momentum=0.0,
                 grad_clip_max_norm:float|None=None, preconditioner_update_probability=1.0, whiten_grad=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner 
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        self._tiny = max([torch.finfo(p.dtype).tiny for p in self._params_with_grad])
        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly 
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p, preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for p in self._params_with_grad]
        self._ms, self._counter_m = None, 0 # momentum buffers and counter  
        self._whiten_grad = whiten_grad # set to False to whiten momentum.  
        if (not whiten_grad) and (self.momentum==0): # expect momentum > 0 when whiten_grad = False
            print("FYI: Set to whiten momentum, but momentum factor is zero.") 


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the Kronecker product gradient/momentum whitening preconditioner.
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)
            
        if self._QLs_exprs is None:
            self._QLs_exprs = [init_kron(g, (torch.mean((torch.abs(g))**4))**(-1/8), 
                                         self._preconditioner_max_size, self._preconditioner_max_skew) for g in grads]
            
        conds = torch.rand(len(grads)) < self.preconditioner_update_probability
        # whitening gradients or momentums     
        if self.momentum==0 or self._whiten_grad: # anyway, needs to the whiten gradient in either case
            pre_grads = [precond_grad_kron_whiten(*QL_exprs, g, self.lr_preconditioner, updateP) 
                         for (QL_exprs, g, updateP) in zip(self._QLs_exprs, grads, conds)]
        
        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                
            if self._whiten_grad: # momentum the whitened gradients 
                [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, pre_grads)]
                pre_grads = self._ms
            else: # whiten the momentum 
                [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
                pre_grads = [precond_grad_kron_whiten(*QL_exprs, m, self.lr_preconditioner, updateP) 
                             for (QL_exprs, m, updateP) in zip(self._QLs_exprs, self._ms, conds)]
        else: # already whitened gradients, just clear the momentum buffers and counter.  
            self._ms, self._counter_m = None, 0              
            
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.sqrt(torch.abs(sum([torch.sum(g*g.conj()) for g in pre_grads]))) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # Update the parameters
        [param.subtract_(lr*g) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        # return whatever closure returns
        return closure_returns
    
    
def update_precond_kron_newton(QL, exprs, V, Hvp, lr=0.01):
    """
    Update the Kron product Newton preconditioner with a pair of vector and hvp, (V, Hvp). 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    Hvp = exprP(*[q.conj() for q in Q], *Q, Hvp) 

    for i, q in enumerate(Q):
        HHc = exprGs[i](Hvp, Hvp.conj())
        VVc = exprGs[i](V, V.conj())
        if q.dim() < 2: # diagonal Q 
            qHHcqh = q * HHc * q.conj()
            qVVcqh = q * VVc * q.conj()
            ell = torch.max(torch.abs(qHHcqh + qVVcqh)) 
            L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
            q.sub_(lr/L[i] * (qHHcqh - qVVcqh) * q)
        else: # matrix Q
            qHHcqh = q @ HHc @ q.H
            qVVcqh = q @ VVc @ q.H
            ell = norm_lower_bound_herm(qHHcqh + qVVcqh) 
            L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
            q.sub_(lr/L[i] * (qHHcqh - qVVcqh) @ q)
    
    if torch.rand([]) < 0.01: # balance factors of Q
        balance_kron_precond(Q)
                    
            
def precond_grad_kron_newton(QL, exprs, G):
    """
    Precondition gradient G with Kron product Newton preconditioner Q. 
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G) 


class KronNewton:
    """
    Implements the Kronecker product Newton-type preconditioner as a class.
    Be extra cautious when using the finite difference method for Hvp approximation (the closure must behave like a pure function). 
    """
    def __init__(self,  params_with_grad, preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale:float|None=None,
                        lr_params=0.01, lr_preconditioner=0.01, momentum=0.0,
                        grad_clip_max_norm:float|None=None, preconditioner_update_probability=1.0,
                        exact_hessian_vector_product=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner        
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        self._tiny = max([torch.finfo(p.dtype).tiny for p in self._params_with_grad])
        self._delta_param_scale = (max([torch.finfo(p.dtype).eps for p in self._params_with_grad])) ** 0.5
        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly 
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._QLs_exprs = [init_kron(p, preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for p in self._params_with_grad]
        self._ms, self._counter_m = None, 0 # momentum buffers and counter 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the Kronecker product Newton-type preconditioner.  
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
                    
                vs = [self._delta_param_scale * torch.randn_like(p) for p in self._params_with_grad]
                [p.add_(v) for (p, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]               
                [p.sub_(v) for (p, v) in zip(self._params_with_grad, vs)] # remove the perturbation            
            
            if self._QLs_exprs is None: # initialize QLs on the fly if it is None 
                self._QLs_exprs = [init_kron(h, (torch.mean((torch.abs(v))**2))**(1/4) * (torch.mean((torch.abs(h))**4))**(-1/8), 
                                             self._preconditioner_max_size, self._preconditioner_max_skew) for (v, h) in zip(vs, Hvs)]
            # update preconditioner
            [update_precond_kron_newton(*QL_exprs, v, h, self.lr_preconditioner) for (QL_exprs, v, h) in zip(self._QLs_exprs, vs, Hvs)]
        else: # only evaluate the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)

        if self.momentum > 0: # precondition the momentum 
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                
            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
            pre_grads = [precond_grad_kron_newton(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else: # precondition the gradient 
            self._ms, self._counter_m = None, 0 # clear the buffer and counter when momentum is set to zero 
            pre_grads = [precond_grad_kron_newton(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            
        if self.grad_clip_max_norm is None: # no grad clipping 
            lr = self.lr_params
        else: # grad clipping 
            grad_norm = torch.sqrt(torch.abs(sum([torch.sum(g*g.conj()) for g in pre_grads]))) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # Update the parameters. 
        [param.subtract_(lr*g) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        # return whatever closure returns
        return closure_returns


#############       End of PSGD Kronecker product preconditioners       #############


#############       Begin of PSGD LRA (low rank approximation) preconditioners       #############


def IpUVtmatvec(U, V, x):
    """
    Returns (I + U*V')*x. All variables are either matrices or column vectors. 
    """
    return x + U.mm(V.t().mm(x))


def update_precond_lra(UVd, Luvd, v, h, lr=0.01, for_whitening=False):
    """
    Update LRA preconditioner Q = (I + U*V')*diag(d) with (vector, Hessian-vector product) = (v, h).
    State variables (U, V, d) and their Lipschitz constant estimates (Lu, Lv, Ld) are updated inplace. 
    When it is used for updating the gradient/momentum whitening preconditioner, we return P*g to save conputations.                  
    U, V, d, v, and h all are either matrices or column vectors.  
    """
    U, V, d = UVd
    Lu, Lv, Ld = Luvd

    Qh = IpUVtmatvec(U, V, d * h)
    if for_whitening: # we return the whitened gradient/momentum if used for updating the whitening preconditioner. 
        Ph = d * IpUVtmatvec(V, U, Qh)

    IpVtU = V.t().mm(U)
    IpVtU.diagonal().add_(1) # avoid forming matrix I explicitly 
    invQtv = v/d
    invQtv = invQtv - V.mm(torch.linalg.solve(IpVtU.t(), U.t().mm(invQtv)))   
    
    a, b = Qh, invQtv
    if torch.rand([]) < 1/3 or U.numel() == 0:
        # Update in the group of diagonal matrices (will impact U and V too). 
        # Note that if U and V are empty, we only need to update d.  
        aa, bb = a * a, b * b
        ell = torch.max(aa + bb)
        Ld.data = torch.max(0.9*Ld + 0.1*ell, ell)
        s = 1 - lr/Ld * (aa - bb)
        U.mul_(s)
        V.div_(s)
        d.mul_(s)
    else: # update U or V
        # Balance the numerical dynamic ranges of U and V. 
        # Not optional as Lu and Lv are not scaling invariant. 
        normU = torch.linalg.vector_norm(U)
        normV = torch.linalg.vector_norm(V)
        rho = torch.sqrt(normU/normV)
        U.div_(rho)
        V.mul_(rho)
    
        if torch.rand([]) < 0.5: # only update U
            atV = a.t().mm(V)
            btV = b.t().mm(V)
            atVVt = atV.mm(V.t())
            btVVt = btV.mm(V.t())
            ell = (torch.linalg.vector_norm(a)*torch.linalg.vector_norm(atVVt) + 
                   torch.linalg.vector_norm(b)*torch.linalg.vector_norm(btVVt))
            Lu.data = torch.max(0.9*Lu + 0.1*ell, ell)
            U.sub_(lr/Lu * ( a.mm(atV.mm(IpVtU)) - b.mm(btV.mm(IpVtU)) ))
        else: # only udate V
            atU = a.t().mm(U)
            btU = b.t().mm(U)
            UUta = U.mm(atU.t())
            UUtb = U.mm(btU.t())
            ell = (torch.linalg.vector_norm(a)*torch.linalg.vector_norm(UUta) + 
                   torch.linalg.vector_norm(b)*torch.linalg.vector_norm(UUtb))
            Lv.data = torch.max(0.9*Lv + 0.1*ell, ell)
            V.sub_(lr/Lv * ( (a + V.mm(atU.t())).mm(atU) - (b + V.mm(btU.t())).mm(btU) ))

    if for_whitening:
        return Ph # return the preconditioned gradient/momentum 
    # else:
    #     return None


def precond_grad_lra(UVd, g):
    """
    Precondition gradient g with Q = (I + U*V')*diag(d).                                      
    All variables here are either matrices or column vectors. 
    """
    U, V, d = UVd
    g = IpUVtmatvec(U, V, d * g)
    g = d * IpUVtmatvec(V, U, g)
    return g


class LRAWhiten:
    """
    Implements the PSGD LRA gradient/momentum whitening preconditioner as a class.
    By default, we whiten the gradient. 
    One can set rank r to zero to get the diagonal preconditioner. 
    """
    def __init__(self,  params_with_grad, rank_of_approximation:int=10, preconditioner_init_scale:float|None=None,
                        lr_params=0.001, lr_preconditioner=0.01, momentum=0.0,
                        grad_clip_max_norm:float|None=None, preconditioner_update_probability=1.0, whiten_grad=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if 2 * rank_of_approximation + 1 >= num_params: # check the rank_of_approximation setting 
            print("FYI: rank r is too high.")
        self._UVd = []
        for _ in range(2):
            # +20 to: 1) avoid /0 when r=0; 2) make sure that norm(U*V') << 1 even with rank_of_approximation=1
            self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device) / (num_params*(rank_of_approximation+20))**0.5)
        if preconditioner_init_scale is None:
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._UVd.append(torch.ones(num_params, 1, dtype=dtype, device=device) * preconditioner_init_scale)
        self._Luvd = [torch.zeros([], dtype=dtype, device=device) for _ in range(3)]
        self._m, self._counter_m = None, 0 # momentum buffer and counter 
        self._whiten_grad = whiten_grad
        if (not whiten_grad) and (self.momentum==0): # expect momentum > 0 when whiten_grad = False
            print("FYI: Set to whiten momentum, but momentum factor is zero.") 


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD LRA gradient/momentum whitening optimizer. 
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)

        # cat grads
        grad = torch.cat([torch.reshape(g, [-1, 1]) for g in grads]) # column vector 
        
        if len(self._UVd) < 3: # initialize d on the fly 
            self._UVd.append((torch.mean(grad**4))**(-1/8) * torch.ones_like(grad)) 

        if self.momentum==0 or self._whiten_grad: # anyway, needs to whiten gradient in either case 
            if torch.rand([]) < self.preconditioner_update_probability:
                pre_grad = update_precond_lra(self._UVd, self._Luvd, torch.randn_like(grad), grad, self.lr_preconditioner, for_whitening=True)
            else:
                pre_grad = precond_grad_lra(self._UVd, grad)
  
        if self.momentum > 0:
            beta = min(self._counter_m/(1 + self._counter_m), self.momentum)
            self._counter_m += 1
            if self._m is None:
                self._m = torch.zeros_like(grad)

            if self._whiten_grad: # whiten gradient 
                self._m.mul_(beta).add_(pre_grad, alpha=1 - beta)
                pre_grad = self._m 
            else: # whiten momentum 
                self._m.mul_(beta).add_(grad, alpha=1 - beta) 
                if torch.rand([]) < self.preconditioner_update_probability:
                    pre_grad = update_precond_lra(self._UVd, self._Luvd, torch.randn_like(self._m), self._m, self.lr_preconditioner, for_whitening=True)
                else:
                    pre_grad = precond_grad_lra(self._UVd, self._m)
        else: # already whitened the gradient; just clear the buffer and counter when momentum is set to zero 
            self._m, self._counter_m = None, 0 
            
        if self.grad_clip_max_norm is None: # no grad clipping 
            lr = self.lr_params
        else: # grad clipping 
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # update the parameters 
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

class LRANewton:
    """
    Implements the PSGD LRA Newton-type preconditioner as a class.
    One can set the rank r to zero to get a diagonal preconditioner. 
    Be extra cautious when using the finite difference method for Hvp approximation (the closure must behave like a pure function).
    """
    def __init__(self,  params_with_grad, rank_of_approximation:int=10, preconditioner_init_scale:float|None=None,
                        lr_params=0.01, lr_preconditioner=0.01, momentum=0.0,
                        grad_clip_max_norm:float|None=None, preconditioner_update_probability=1.0,
                        exact_hessian_vector_product=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps**0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if 2 * rank_of_approximation + 1 >= num_params: # check the rank_of_approximation setting 
            print("FYI: rank r is too high.")
        self._UVd = []
        for _ in range(2):
            # +20 to: 1) avoid /0; 2) make sure that norm(U*V') << 1 even when rank_of_approximation=1
            self._UVd.append(torch.randn(num_params, rank_of_approximation, dtype=dtype, device=device) / (num_params*(rank_of_approximation+20))**0.5)
        if preconditioner_init_scale is None:
            print("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
        else:
            self._UVd.append(torch.ones(num_params, 1, dtype=dtype, device=device) * preconditioner_init_scale)
        self._Luvd = [torch.zeros([], dtype=dtype, device=device) for _ in range(3)]
        self._m, self._counter_m = None, 0 # momentum buffer and counter 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of the PSGD LRA Newton optimizer. 
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
                
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
                [param.sub_(v) for (param, v) in zip(self._params_with_grad, vs)]

            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) # column vector
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) # column vector  
            if len(self._UVd) < 3: # init d if not in the UVd list 
                self._UVd.append((torch.mean(v*v))**(1/4) * (torch.mean(h**4))**(-1/8) * torch.ones_like(v))
            
            # update preconditioner
            update_precond_lra(self._UVd, self._Luvd, v, h, self.lr_preconditioner)
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
            
        if self.grad_clip_max_norm is None: # no grad clipping 
            lr = self.lr_params
        else: # clip grad 
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # update the parameters
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

#############       End of PSGD LRA preconditioners       #############


#############       Begin of PSGD dense matrix Newton-type preconditioner       #############


class DenseNewton:
    """
    Implements the PSGD dense matrix Newton-type preconditioner as a class. 
    Be extra cautious when using the finite difference method for Hvp approximation (the closure must behave like a pure function).
    It's mainly for illustrating how PSGD works due to its simplicity. 
    It's also a good alternative of the BFGS like quasi-Newton methods as no line search is required. 
    """
    def __init__(self, params_with_grad, preconditioner_init_scale:float|None=None,
                 lr_params=0.01, lr_preconditioner=0.01, momentum=0.0, 
                 grad_clip_max_norm:float|None=None, preconditioner_update_probability=1.0,
                 exact_hessian_vector_product=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad]  # double check requires_grad flag
        dtype, device = self._params_with_grad[0].dtype, self._params_with_grad[0].device
        self._tiny = torch.finfo(dtype).tiny
        self._delta_param_scale = torch.finfo(dtype).eps ** 0.5
        self._param_sizes = [torch.numel(param) for param in self._params_with_grad]
        self._param_cumsizes = torch.cumsum(torch.tensor(self._param_sizes), 0)
        num_params = self._param_cumsizes[-1]
        if preconditioner_init_scale is None: # initialize Q on the fly
            self._Q = None 
        else:
            self._Q = torch.eye(num_params, dtype=dtype, device=device) * preconditioner_init_scale
        self._L = torch.zeros([]) # Lipschitz constant estimation for the psgd criterion 
        self._m, self._counter_m = None, 0 # buffer and counter for momentum 
        self._exact_hessian_vector_product = exact_hessian_vector_product
        if not exact_hessian_vector_product:
            print("FYI: Approximate Hvp with finite-difference method. Make sure that: 1) the closure behaves like a pure function; 2) delta param scale is proper.")


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with dense matrix Newton-type preconditioner. 
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
                
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]
                [param.add_(v) for (param, v) in zip(self._params_with_grad, vs)]
                with torch.enable_grad():
                    perturbed_returns = closure()
                    perturbed_loss = perturbed_returns if isinstance(perturbed_returns, torch.Tensor) else perturbed_returns[0]
                    perturbed_grads = torch.autograd.grad(perturbed_loss, self._params_with_grad)
                Hvs = [perturbed_g - g for (perturbed_g, g) in zip(perturbed_grads, grads)]
                vs = [self._delta_param_scale * torch.randn_like(param) for param in self._params_with_grad]

            v = torch.cat([torch.reshape(v, [-1, 1]) for v in vs]) 
            h = torch.cat([torch.reshape(h, [-1, 1]) for h in Hvs]) 
            if self._Q is None: # initialize Q on the fly if it is None
                self._Q = torch.eye(len(v), dtype=v.dtype, device=v.device) * (torch.mean(v*v))**(1/4) * (torch.mean(h**4))**(-1/8)

            # this is how psgd updates Q
            a = self._Q @ (self._Q.T @ (self._Q @ h))
            b = self._Q @ v
            ell = torch.sum(a*a + b*b)
            self._L = torch.max(0.9*self._L + 0.1*ell, ell)
            self._Q = self._Q - self.lr_preconditioner/self._L * (a @ (a.T @ self._Q) - b @ (b.T @ self._Q)) 
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
            pre_grad = self._Q.T @ (self._Q @ self._m)
        else:
            self._m, self._counter_m = None, 0 # clear the buffer and counter when momentum is set to zero 
            pre_grad = self._Q.T @ (self._Q @ grad)
        
        if self.grad_clip_max_norm is None: # no grad clipping 
            lr = self.lr_params
        else: # grad clipping 
            grad_norm = torch.linalg.vector_norm(pre_grad) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm / grad_norm, 1.0)

        # update the parameters
        [param.subtract_(lr * pre_grad[j - i:j].view_as(param)) 
         for (param, i, j) in zip(self._params_with_grad, self._param_sizes, self._param_cumsizes)]
        
        # return whatever closure returns
        return closure_returns
    

#############       End of PSGD dense matrix Newton-type preconditioner       #############
