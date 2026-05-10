"""
We implement a simplest dependency-free PSGD Kron momentum whitening optimizer:
    * Only consider 0/1/2D momentum whitening with real bfloat16 preconditioners. 
    * Higher order tensors are matricized (you can redefine _matricize()). 
    * Always diag preconditioner for 0/1D tensors; diag/matrix preconditioner for 2D tensors.  
"""

import torch


def norm_lower_bound_spd(A, k=128, half_iters=2):
    """
    A simplified version of psgd.norm_lower_bound_spd with plain random init (no centroid alignment). 
    """
    normalizing_factor = A.diagonal().amax() + 2**-126
    A = A / normalizing_factor 
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(half_iters):
        V = V @ A 
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + 2**-126
        V = V @ A   
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))


def norm_lower_bound_skh(A, k=128, half_iters=2):
    """
    A simplified version psgd.norm_lower_bound_skh with plain random init (no centroid alignment). 
    """
    normalizing_factor = A.abs().amax() + 2**-126
    A = A / normalizing_factor  
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    for _ in range(half_iters):
        V = V @ A 
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + 2**-126
        V = V @ A   
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))


def procrustes_step2(Q, max_step_size=1/8):
    """
    A simplified version of psgd.procrustes_step2 just for real matrices. 
    """
    R = Q.T - Q 
    R /= norm_lower_bound_skh(R) + 2**-126 
    RQ = R @ Q
    RRQ = R @ RQ
    tr_RQ = RQ.diagonal().sum()
    tr_RRQ = RRQ.diagonal().sum() 
    a = torch.where(tr_RRQ < 0, torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size), max_step_size)
    Q.add_(a * (RQ + 0.5 * a * RRQ))


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0):
    """
    A simplified version of psgd.init_kron: only dQ="Q0.5EQ1.5"; always diag Q for 0/1D tensor. 
    """
    shape = t.shape
    if len(shape) not in [0, 1, 2]:
        raise ValueError(f"Only 0D, 1D and 2D param supported; got shape {shape}.")
    
    if len(shape) <= 1:
        Q = [Scale * torch.ones(shape, dtype=t.dtype, device=t.device)]
        L = [torch.zeros([], dtype=torch.float32, device=t.device)]
        return [Q, L]
    
    scale = Scale ** 0.5
    Q, L = [], []
    for size in shape:
        L.append(torch.zeros([], dtype=torch.float32, device=t.device))
        if size <= 1 or size > max_size or size * size > max_skew * t.numel():
            Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
        else:
            Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))
    return [Q, L]


def _balance_2(Ql, Qr):
    """
    A simplified version of psgd.balance_kron_precond. 
    """
    rho = (Qr.abs().amax() / Ql.abs().amax()).sqrt()
    Ql.mul_(rho)
    Qr.div_(rho)


def update_diag(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for diag preconditioner.
    """
    Q, L = QL
    Q0 = Q[0]

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G)

    Pg = (Q0 * Q0) * Gd

    term1 = Pg * Pg
    ell = term1.amax() + 1 # term2 = total_numel / Q0.numel() = 1
    L[0].copy_(torch.max(betaL * L[0] + (1 - betaL) * ell, ell))
    Q0.mul_(1 - lr / L[0] * (term1 - 1))


def update_dense_dense(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for kron(dense, dense) preconditioner.
    """
    Q, L = QL
    Ql, Qr = Q
    m, n = G.shape

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G)
    Pg = Ql @ Gd @ Qr.T
    Pg = Ql.T @ Pg @ Qr

    term1 = Pg @ Pg.T
    ell = norm_lower_bound_spd(term1) + n
    L[0].copy_(torch.max(betaL * L[0] + (1 - betaL) * ell, ell))
    term1.diagonal().sub_(n)
    Ql.sub_(lr / L[0] * (term1 @ Ql))
    procrustes_step2(Ql)

    term1 = Pg.T @ Pg
    ell = norm_lower_bound_spd(term1) + m
    L[1].copy_(torch.max(betaL * L[1] + (1 - betaL) * ell, ell))
    term1.diagonal().sub_(m)
    Qr.sub_(lr / L[1] * (term1 @ Qr))
    procrustes_step2(Qr)

    if torch.rand([]) < 0.01:
        _balance_2(Ql, Qr)


def update_dense_diag(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for kron(diag, dense) preconditioner.
    """
    Q, L = QL
    Ql, Qr = Q
    m, n = G.shape

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G)
    Pg = Ql.T @ (Ql @ Gd)
    Pg = Pg * (Qr * Qr)

    term1 = Pg @ Pg.T
    ell = norm_lower_bound_spd(term1) + n
    L[0].copy_(torch.max(betaL * L[0] + (1 - betaL) * ell, ell))
    term1.diagonal().sub_(n)
    Ql.sub_(lr / L[0] * (term1 @ Ql))
    procrustes_step2(Ql)

    term1 = (Pg * Pg).sum(dim=0)
    ell = term1.amax() + m
    L[1].copy_(torch.max(betaL * L[1] + (1 - betaL) * ell, ell))
    Qr.mul_(1 - lr / L[1] * (term1 - m))

    if torch.rand([]) < 0.01:
        _balance_2(Ql, Qr)


def update_diag_dense(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for kron(dense, diag) preconditioner.
    """
    Q, L = QL
    Ql, Qr = Q
    m, n = G.shape

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G)
    Pg = (Gd @ Qr.T) @ Qr
    Pg = (Ql * Ql).unsqueeze(1) * Pg

    term1 = (Pg * Pg).sum(dim=1)
    ell = term1.amax() + n
    L[0].copy_(torch.max(betaL * L[0] + (1 - betaL) * ell, ell))
    Ql.mul_(1 - lr / L[0] * (term1 - n))

    term1 = Pg.T @ Pg
    ell = norm_lower_bound_spd(term1) + m
    L[1].copy_(torch.max(betaL * L[1] + (1 - betaL) * ell, ell))
    term1.diagonal().sub_(m)
    Qr.sub_(lr / L[1] * (term1 @ Qr))
    procrustes_step2(Qr)

    if torch.rand([]) < 0.01:
        _balance_2(Ql, Qr)


def update_diag_diag(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for kron(diag, diag) preconditioner.
    """
    Q, L = QL
    Ql, Qr = Q
    m, n = G.shape

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G)
    Pg = Gd * (Qr * Qr) * (Ql * Ql).unsqueeze(1)
    Pg2 = Pg * Pg

    term1 = Pg2.sum(dim=1)
    ell = term1.amax() + n
    L[0].copy_(torch.max(betaL * L[0] + (1 - betaL) * ell, ell))
    Ql.mul_(1 - lr / L[0] * (term1 - n))

    term1 = Pg2.sum(dim=0)
    ell = term1.amax() + m
    L[1].copy_(torch.max(betaL * L[1] + (1 - betaL) * ell, ell))
    Qr.mul_(1 - lr / L[1] * (term1 - m))

    if torch.rand([]) < 0.01:
        _balance_2(Ql, Qr)


def apply_diag(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for diag preconditioner. 
    """
    Q0 = QL[0][0]
    return (Q0 * Q0) * G


def apply_dense_dense(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for kron(dense, dense) preconditioner. 
    """
    Ql, Qr = QL[0]
    Pg = Ql @ G @ Qr.T
    return Ql.T @ Pg @ Qr


def apply_dense_diag(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for kron(diag, dense) preconditioner. 
    """
    Ql, Qr = QL[0]
    Pg = Ql.T @ (Ql @ G)
    return Pg * (Qr * Qr)


def apply_diag_dense(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for kron(dense, diag) preconditioner. 
    """
    Ql, Qr = QL[0]
    Pg = (G @ Qr.T) @ Qr
    return (Ql * Ql).unsqueeze(1) * Pg


def apply_diag_diag(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for kron(diag, diag) preconditioner. 
    """
    Ql, Qr = QL[0]
    return G * (Qr * Qr) * (Ql * Ql).unsqueeze(1)


def _dispatch(
        Q, 
        _table={
            (2, 2): (update_dense_dense, apply_dense_dense),
            (2, 1): (update_dense_diag, apply_dense_diag),
            (1, 2): (update_diag_dense, apply_diag_dense),
            (1, 1): (update_diag_diag, apply_diag_diag),}
            ):
    """
    Picks (update_fn, apply_fn) pair based on factor dims. No einsum exprs saved.
    As the name suggests, do not mutate _table.  
    """
    if len(Q) == 1:
        return (update_diag, apply_diag)
    
    return _table[(Q[0].dim(), Q[1].dim())]


def _matricize(grad):
    """
    First squeeze out singleton axes. Then:
        reshape a >2D tensor to 2D by the split that's closest to square;
        do nothing for <=2D tensor. 
    Feel free to redefine this function if you want different behaviors, e.g.,
        Reshape [d0, d1, d2, ...] to [d0, d1 * d2 * ...];
        Reshape 1D vector to [1, d0] or [d0, 1] (if you want dense-Q on vector). 
    """
    grad = grad.squeeze()
    if grad.dim() <= 2:
        return grad
    
    shape = grad.shape
    total = grad.numel()
    best_k, best_ratio, left = 1, float("inf"), 1
    for k in range(1, len(shape)):
        left *= shape[k - 1]
        right = total // left
        ratio = max(left, right) / min(left, right)
        if ratio < best_ratio:
            best_ratio, best_k = ratio, k

    new_left = 1
    for i in range(best_k):
        new_left *= shape[i]

    return grad.reshape(new_left, -1)


class KWNS4(torch.optim.Optimizer):
    """
    A simplified version of wrapped_as_torch_optimizer_for_ddp.py (a DDP wrapping). 
    Important tips:
        initial value lr_preconditioner=0.5 is too high and needs to anneal to ~0.1;
        initial value preconditioner_update_probability=1.0 is too high and needs to anneal to 0.01~0.1.  
    """
    def __init__(
            self,
            params,
            preconditioner_max_size=float("inf"), 
            preconditioner_max_skew=1.0, # for 2D tensor, 0.0 => all diagonal Q; inf => all dense Q
            preconditioner_init_scale=1.0, # P0 = preconditioner_init_scale^2 * I; set to smaller values if unsure
            lr_params=3e-4, 
            lr_preconditioner=0.5, # Quickly anneal down to ~ 0.1; don't anneal to ~ 0.01 as eps(bf16) ~ 0.01    
            betaL=0.9, 
            damping=1e-9, # roughly the eps in Adam(W) 
            momentum=0.9, # roughly the beta1 in Adam(W)
            weight_decay=0.0, 
            decoupled_weight_decay=True, # True for decoupled weight decay; False for the classic weight decay  
            grad_clip_max_amps=(2.0, 10.0), # clip grad with thresholds (max average amplitude, max element-wise amplitude) 
            preconditioner_update_probability=1.0, # Quickly anneal to 0.01 ~ 0.1 to save computations
            resync_every=1_000_000, # resync every # steps if nondeterministic matmul diverges states too much; generally no need.   
    ):
        assert preconditioner_max_size >= 0.0
        assert preconditioner_max_skew >= 0.0
        assert preconditioner_init_scale > 0.0
        assert lr_params > 0.0
        assert 0.0 < lr_preconditioner < 1.0
        assert 0.0 <= betaL <= 1.0
        assert damping >= 0.0
        assert 0.0 <= momentum < 1.0
        assert weight_decay >= 0.0
        assert isinstance(decoupled_weight_decay, bool)
        assert grad_clip_max_amps[1] >= grad_clip_max_amps[0] >= 1.0 
        assert 0.0 < preconditioner_update_probability <= 1.0
        assert resync_every > 0

        defaults = {
            "preconditioner_max_size": preconditioner_max_size, 
            "preconditioner_max_skew": preconditioner_max_skew,
            "preconditioner_init_scale": preconditioner_init_scale,
            "lr_params": lr_params,  
            "lr_preconditioner": lr_preconditioner, 
            "betaL": betaL, 
            "damping": damping, 
            "momentum": momentum,
            "weight_decay": weight_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "grad_clip_max_amps": grad_clip_max_amps, 
            "preconditioner_update_probability": preconditioner_update_probability,
            "resync_every": resync_every,
        }
        super().__init__(params, defaults)

        self._step = 0

        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed: # if True, assume multi-GPU DDP training; important to sync the rng states
            state = torch.get_rng_state().cuda() # assume nccl backend
            torch.distributed.broadcast(state, src=0)
            self.cpu_rng_state = state.cpu()

            state = torch.cuda.get_rng_state().cuda() # assume nccl backend 
            torch.distributed.broadcast(state, src=0)
            self.cuda_rng_state = state.cpu()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.is_distributed: # sync internal rng states; save external rng states 
            external_cpu_rng_state = torch.get_rng_state()
            external_cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(self.cpu_rng_state)
            torch.cuda.set_rng_state(self.cuda_rng_state)

        for group in self.param_groups:
            momentum = group["momentum"]
            max_avg_amp, max_element_amp = group["grad_clip_max_amps"]
            prb = group["preconditioner_update_probability"]
            update_P = int(self._step * prb + 1) > int((self._step - 1) * prb + 1) # +1 so that update_P=True for step=0
                
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                wd = group["weight_decay"]
                if wd > 0.0: 
                    if group["decoupled_weight_decay"]:
                        p.mul_(1.0 - wd * group["lr_params"])
                    else:
                        grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0: # initialization
                    grad = _matricize(grad)
                    state["ema"] = torch.zeros_like(grad, dtype=p.dtype)
                    state["QL"] = init_kron(grad.to(torch.bfloat16), 
                                            Scale=group["preconditioner_init_scale"], 
                                            max_size=group["preconditioner_max_size"], 
                                            max_skew=group["preconditioner_max_skew"])
                    state["step"] = 0
                else:
                    grad = grad.reshape(state["ema"].shape)

                update_fn, apply_fn = _dispatch(state["QL"][0])

                t = state["step"]
                beta = min(t/(t + 1), momentum)
                state["ema"].mul_(beta).add_(grad, alpha=1.0 - beta) # state["ema"].lerp_(grad, 1.0 - beta)
                state["step"] += 1

                ema_bf16 = state["ema"].to(torch.bfloat16)

                if update_P:
                    update_fn(state["QL"], ema_bf16, 
                              lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])

                h = apply_fn(state["QL"], ema_bf16)

                avg_amp = torch.sqrt(torch.mean(h * h))
                h *= torch.clamp(max_avg_amp/avg_amp, max=1.0) # ok with avg_amp = 0.0
                h.clamp_(min=-max_element_amp, max=max_element_amp) 
                p.subtract_(h.view_as(p), alpha=group["lr_params"])

                # resync states occasionally if matmul is not deterministic and state divergence is large 
                if self.is_distributed and (state["step"] % group["resync_every"] == 0):
                    torch.distributed.broadcast(p, src=0)
                    torch.distributed.broadcast(state["ema"], src=0)
                    for q, ell in zip(*state["QL"]):
                        torch.distributed.broadcast(q, src=0)
                        torch.distributed.broadcast(ell, src=0)

        if self.is_distributed: # save internal rng states; recover external rng states 
            self.cpu_rng_state = torch.get_rng_state()
            self.cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(external_cpu_rng_state)
            torch.cuda.set_rng_state(external_cuda_rng_state)

        self._step += 1
        return loss

