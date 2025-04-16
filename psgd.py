import torch
import opt_einsum


def norm_lower_bound_herm(A):
    """
    Returns a cheap lower bound for the spectral norm of a symmetric or Hermitian matrix A.
    """
    max_abs = torch.max(torch.abs(A)) # used to normalize A to avoid numerically under- or over-flow
    if max_abs > 0:
        A = A/max_abs
        aa = torch.real(A * A.conj())
        _, j = torch.max(torch.sum(aa, dim=1), 0)
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))
    else: # must have A=0
        return max_abs 
    
    
###############################################################################
#               The Kronecker product preconditioner


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=float("inf")):
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
            else:
                # use matrix as preconditioner for this dim 
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


def precond_grad_kron_whiten(QL, exprs, G, lr=0.02, updateP=True):
    """
    Precondition gradient G with preconditioner Q. 
    We just update the preconditioner here to save some computations. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    G = exprP(*[q.conj() for q in Q], *Q, G) 
    
    if updateP:
        order = G.dim() # order of tensor 
        if order>1 and torch.rand([])<0.01:
            # balance the dynamic range of Q if there are more than one factors 
            norms = [torch.max(torch.abs(q)) for q in Q]
            gmean = (torch.cumprod(torch.stack(norms), dim=0)[-1])**(1/order) # geometric mean 
            for i, q in enumerate(Q):
                q.mul_(gmean/norms[i]) 
    
        total_numel = G.numel() 
        for i, q in enumerate(Q):
            GGc = exprGs[i](G, G.conj())
            if q.dim() < 2:
                target_scale = total_numel/q.numel()
                ell = torch.max(torch.abs(GGc)) + target_scale
                L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
                q.sub_(lr/L[i] * (GGc - target_scale) * q)
            else:
                target_scale = total_numel/q.shape[0]
                ell = norm_lower_bound_herm(GGc) + target_scale
                L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
                q.sub_(lr/L[i] * (GGc @ q - target_scale*q))
                    
    return G
                    

class KronWhiten:
    """
    Implements the Kronecker Whitening product preconditioner.
    Propose to set 0.01 < lr_preconditioner < 0.1.  
    By default, we whiten the gradient, not the momentum. 
    """
    def __init__(self,  params_with_grad, 
                 preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale=None,
                 lr_params=0.001, lr_preconditioner=0.02, momentum=0.0,
                 grad_clip_max_norm=None, preconditioner_update_probability=1.0, whiten_grad=True):
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
            print("FYI: Will set the preconditioner initial scale on the fly. Highly recommend to set it manually!")
        else:
            self._QLs_exprs = [init_kron(p, preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for p in self._params_with_grad]
        self._ms = None # momentum buffers 
        self._counter = 0 # counter for momentum 
        self._whiten_grad = whiten_grad # set to False to whiten momentum. 


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the Kronecker product whitening preconditioner.
        """
        with torch.enable_grad():
            closure_returns = closure()
            loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
            grads = torch.autograd.grad(loss, self._params_with_grad)
            
        if self._QLs_exprs is None:
            self._QLs_exprs = [init_kron(g, (torch.mean((torch.abs(g))**4))**(-1/8), 
                                         self._preconditioner_max_size, self._preconditioner_max_skew) for g in grads]
            
        # preconditioned gradients; momentum is optional    
        updateP = torch.rand([]) < self.preconditioner_update_probability
        if self.momentum > 0:
            beta = min(self._counter/(1 + self._counter), self.momentum)
            self._counter += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                
            if self._whiten_grad: # whiten the gradient 
                pre_grads = [precond_grad_kron_whiten(*QL_exprs, g, self.lr_preconditioner, updateP) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
                [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, pre_grads)]
                pre_grads = self._ms
            else: # whiten the momentum 
                [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
                pre_grads = [precond_grad_kron_whiten(*QL_exprs, m, self.lr_preconditioner, updateP) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else:
            self._ms, self._counter = None, 0 # clean the buffer and counter when momentum is set to zero 
            pre_grads = [precond_grad_kron_whiten(*QL_exprs, g, self.lr_preconditioner, updateP) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.sqrt(torch.abs(sum([torch.sum(g*g.conj()) for g in pre_grads]))) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # Update the parameters.
        [param.subtract_(lr*g) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        # return whatever closure returns
        return closure_returns
    
    
def update_precond_kron_newton(QL, exprs, V, Hvp, lr=0.02):
    """
    Update the Kron product Newton preconditioner with a pair of vector and hvp (V, Hvp). 
    Must have V ~ N(0, 1) in this implementation. 
    """   
    Q, L = QL
    exprP, exprGs = exprs
    Hvp = exprP(*[q.conj() for q in Q], *Q, Hvp) 
    
    order = Hvp.dim() # order of tensor 
    if order>1 and torch.rand([])<0.01:
        # balance the dynamic range of Q if there are more than one factors 
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = (torch.cumprod(torch.stack(norms), dim=0)[-1])**(1/order) # geometric mean 
        for i, q in enumerate(Q):
            q.mul_(gmean/norms[i]) 

    total_numel = Hvp.numel() 
    for i, q in enumerate(Q):
        HHc = exprGs[i](Hvp, Hvp.conj())
        VVc = exprGs[i](V, V.conj())
        if q.dim() < 2:
            ell = torch.max(torch.abs(HHc)) + total_numel/q.numel()
            L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
            q.sub_(lr/L[i] * (HHc - VVc) * q)
        else:
            ell = norm_lower_bound_herm(HHc) + total_numel/q.shape[0]
            L[i].data = torch.max(0.9*L[i] + 0.1*ell, ell)
            q.sub_(lr/L[i] * (HHc - VVc) @ q)
                    
            
def precond_grad_kron_newton(QL, exprs, G):
    """
    Precondition gradient G with Kron product Newton preconditioner Q. 
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G) 


class KronNewton:
    """
    Implements the Kronecker product Newton preconditioner, Q = sum_i kron(Qi1, Qi2, Qi3, ...), as a class.
    Propose to set 0.01 < lr_preconditioner < 0.1. 
    Be extra cautious when using the finite difference method for Hvp approximation (the closure must behave like a pure function). 
    """
    def __init__(self,  params_with_grad, preconditioner_max_size=float("inf"), preconditioner_max_skew=1.0, preconditioner_init_scale=None,
                        lr_params=0.01, lr_preconditioner=0.02, momentum=0.0,
                        grad_clip_max_norm=None, preconditioner_update_probability=1.0,
                        exact_hessian_vector_product:bool=True):
        # mutable members
        self.lr_params = lr_params
        self.lr_preconditioner = lr_preconditioner        
        self.momentum = momentum if (0<momentum<1) else 0.0
        self.grad_clip_max_norm = grad_clip_max_norm
        self.preconditioner_update_probability = preconditioner_update_probability
        # protected members
        self._exact_hessian_vector_product = exact_hessian_vector_product
        self._preconditioner_max_size = preconditioner_max_size
        self._preconditioner_max_skew = preconditioner_max_skew
        params_with_grad = [params_with_grad,] if isinstance(params_with_grad, torch.Tensor) else params_with_grad
        self._params_with_grad = [param for param in params_with_grad if param.requires_grad] # double check requires_grad flag 
        self._tiny = max([torch.finfo(p.dtype).tiny for p in self._params_with_grad])
        self._delta_param_scale = (max([torch.finfo(p.dtype).eps for p in self._params_with_grad])) ** 0.5
        if preconditioner_init_scale is None:
            self._QLs_exprs = None # initialize on the fly 
            print("FYI: Will set the preconditioner initial scale on the fly. Highly recommend to set it manually!")
        else:
            self._QLs_exprs = [init_kron(p, preconditioner_init_scale, preconditioner_max_size, preconditioner_max_skew) for p in self._params_with_grad]
        self._ms = None # momentum buffers 
        self._counter = 0 # counter for momentum 


    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single step of PSGD with the Kronecker product Newton preconditioner.  
        """
        if (torch.rand([]) < self.preconditioner_update_probability) or (self._QLs_exprs is None):
            # evaluates gradients, Hessian-vector product, and updates the preconditioner
            if self._exact_hessian_vector_product:
                # exact Hessian-vector product
                with torch.enable_grad():
                    closure_returns = closure()
                    loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                    grads = torch.autograd.grad(loss, self._params_with_grad, create_graph=True)
                    vs = [torch.randn_like(p) for p in self._params_with_grad]
                    Hvs = torch.autograd.grad(grads, self._params_with_grad, vs) # this line also works for complex matrices 
            else:
                # approximate the Hessian-vector product via finite-difference formulae. Use it with cautions.
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
                
                [p.sub_(v) for (p, v) in zip(self._params_with_grad, vs)] # simpler to remove perturbation here (Yang Gao)             
                [v.div_(self._delta_param_scale) for v in vs]
                [h.div_(self._delta_param_scale) for h in Hvs]
            
            # update preconditioner 
            # initialize QLs if it is None 
            if self._QLs_exprs is None:
                self._QLs_exprs = [init_kron(h, (torch.mean((torch.abs(v))**2))**(1/4) * (torch.mean((torch.abs(h))**4))**(-1/8), 
                                             self._preconditioner_max_size, self._preconditioner_max_skew) for (v, h) in zip(vs, Hvs)]
            # update QLs
            [update_precond_kron_newton(*QL_exprs, v, h, self.lr_preconditioner) for (QL_exprs, v, h) in zip(self._QLs_exprs, vs, Hvs)]
        else:
            # only evaluates the gradients
            with torch.enable_grad():
                closure_returns = closure()
                loss = closure_returns if isinstance(closure_returns, torch.Tensor) else closure_returns[0]
                grads = torch.autograd.grad(loss, self._params_with_grad)

        # preconditioned gradients; momentum is optional      
        if self.momentum > 0:
            beta = min(self._counter/(1 + self._counter), self.momentum)
            self._counter += 1
            if self._ms is None:
                self._ms = [torch.zeros_like(g) for g in grads]
                
            [m.mul_(beta).add_(g, alpha=1 - beta) for (m, g) in zip(self._ms, grads)]
            pre_grads = [precond_grad_kron_newton(*QL_exprs, m) for (QL_exprs, m) in zip(self._QLs_exprs, self._ms)]
        else:
            self._ms, self._counter = None, 0 # clean the buffer and counter when momentum is set to zero 
            pre_grads = [precond_grad_kron_newton(*QL_exprs, g) for (QL_exprs, g) in zip(self._QLs_exprs, grads)]
            
        # gradient clipping is optional
        if self.grad_clip_max_norm is None:
            lr = self.lr_params
        else:
            grad_norm = torch.sqrt(torch.abs(sum([torch.sum(g*g.conj()) for g in pre_grads]))) + self._tiny
            lr = self.lr_params * min(self.grad_clip_max_norm/grad_norm, 1.0)
            
        # Update the parameters. 
        [param.subtract_(lr*g) for (param, g) in zip(self._params_with_grad, pre_grads)]
        
        # return whatever closure returns
        return closure_returns


################## end of the Kronecker product preconditioner #################################


