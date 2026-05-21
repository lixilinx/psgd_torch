"""
Simple dependency-free Kron momentum Whitening optimizer with NS iterations for inv 4th root of E[gg^T] (KWNS4).
Largely corresponds to psgd.KronWhiten with dQ=Q0p5EQ1p5, but with some adaptations:
    * Only real bfloat16 momentun whitening preconditioners. 
    * Always diag preconditioner for 0/1D tensors; per-axis diag/matrix preconditioner for >=2D tensors.
    * Einsum is replaced with matmul and multi_dot to avoid any overhead.   
    * You can redefine _tensorize(), e.g., 1) merge small adjacent dims; 2) reshape vector to matrix to use dense preconditioner.   
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
    A simplified version of psgd.norm_lower_bound_skh with plain random init (no centroid alignment). 
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
    tr_RQ = RQ.diagonal().sum() # trace not implemented for CPU bf16 matrix
    tr_RRQ = RRQ.diagonal().sum() 
    a = torch.where(tr_RRQ < 0, torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size), max_step_size)
    Q.add_(a * (RQ + 0.5 * a * RRQ))


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0):
    """
    A simplified version of psgd.init_kron: only for dQ="Q0.5EQ1.5"; no einsum exprs. 
    """
    shape = t.shape
    
    if len(shape) <= 1:
        Q = [Scale * torch.ones(shape, dtype=t.dtype, device=t.device)]
        L = [torch.zeros([], dtype=torch.float32, device=t.device)]
        return [Q, L]
    
    scale = Scale ** (1.0 / len(shape))
    Q, L = [], []
    for size in shape:
        L.append(torch.zeros([], dtype=torch.float32, device=t.device))
        if size <= 1 or size > max_size or size * size > max_skew * t.numel():
            Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))
        else:
            Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))
    return [Q, L]


def _balance_n(Q):
    """
    A simplified version of psgd.balance_kron_precond. 
    """
    norms = [q.abs().amax() for q in Q]
    gmean = torch.prod(torch.stack(norms)) ** (1.0 / len(Q))
    for q, nrm in zip(Q, norms):
        q.mul_(gmean / nrm)


def update_diag(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 for diag preconditioner.
    """
    Q, L = QL
    Q0 = Q[0]

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G) # 2**-8 = eps(bf16)/2

    Pg = Q0 * Q0 * Gd

    term1 = Pg * Pg
    ell = term1.amax() + 1 # term2 = total_numel / Q0.numel() = 1
    L[0].mul_(betaL).add_(ell, alpha=1 - betaL).clamp_(min=ell) # L[0].copy_(torch.maximum(betaL * L[0] + (1 - betaL) * ell, ell))
    Q0.mul_(1 - lr / L[0] * (term1 - 1))


def update_kron(QL, G, lr=0.1, betaL=0.9, damping=1e-9):
    """
    A plain implementation of psgd.update_precond_kron_whiten_q0p5eq1p5 with matmul, no einsum.
    Unlike psgd.update_precond_kron_whiten_q0p5eq1p5, update_kron here only works for >=2D tensors. 
    """
    Q, L = QL
    N = G.dim()
    total_numel = G.numel()

    Gd = G + (damping + 2**-8 * G.abs()) * torch.randn_like(G) # 2**-8 = eps(bf16)/2
    Pg = apply_kron(QL, Gd)
    for i, q in enumerate(Q):
        n_i = q.shape[0]
        term2 = total_numel / n_i
        others = [d for d in range(N) if d != i]
        if q.dim() < 2: 
            term1 = torch.linalg.vector_norm(Pg, dim=others).square_() # (Pg * Pg).sum(dim=others) 
            ell = term1.amax() + term2  
            L[i].mul_(betaL).add_(ell, alpha=1 - betaL).clamp_(min=ell) # L[i].copy_(torch.maximum(betaL * L[i] + (1 - betaL) * ell, ell))
            q.mul_(1 - lr / L[i] * (term1 - term2))
        else:
            flat = Pg.movedim(i, 0).reshape(n_i, -1)
            term1 = flat @ flat.T 
            ell = norm_lower_bound_spd(term1) + term2
            L[i].mul_(betaL).add_(ell, alpha=1 - betaL).clamp_(min=ell) # L[i].copy_(torch.maximum(betaL * L[i] + (1 - betaL) * ell, ell))
            term1.diagonal().sub_(term2)
            q.sub_(lr / L[i] * (term1 @ q))
            procrustes_step2(q)


def apply_diag(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron for diag preconditioner. 
    """
    Q0 = QL[0][0]
    return Q0 * Q0 * G


def apply_kron(QL, G):
    """
    A plain implementation of psgd.precond_grad_kron with matmul, no einsum. 
    No universal best einsum implementation. The one here is simple and not bad.  
    """
    Q = QL[0]
    N = G.dim()
    Pg = G
    for i, q in enumerate(Q):
        if q.dim() < 2:
            s = q * q
            Pg = Pg * s.view([-1] + [1] * (N - i - 1))
        else:
            n_i = q.shape[0]
            if i < N - 1:
                Pg = Pg.movedim(i, 0) # for row-major data, moving i=>0 is cheaper than i=>-1
                flat = Pg.reshape(n_i, -1)
                flat = torch.linalg.multi_dot([q.T, q, flat])
                Pg = flat.view_as(Pg).movedim(0, i)
            else: # apply P on the last axis directly to save memory copy
                flat = Pg.reshape(-1, n_i)
                flat = torch.linalg.multi_dot([flat, q.T, q])
                Pg = flat.view_as(Pg)
    return Pg
        

def _dispatch(Q):
    """
    Picks (update_fn, apply_fn) pair based on factor dims. No einsum exprs saved.
    """
    if len(Q) == 1:
        return (update_diag, apply_diag)
    return (update_kron, apply_kron)


def _tensorize(grad):
    """
    Feel free to redefine this function if you want different behaviors, e.g.,
        Merge small adjacent dims, say [64, 32, 3, 3] to [64, 32, 9];
        Reshape 1D vector to [1, d] or [d, 1] if you want dense-Q on vector. 
    Do not use transpose, permute and movedim (otherwise, you need an _inverse_tensorize()). 
    """
    return grad.squeeze()


class KWNS4(torch.optim.Optimizer):
    """
    A simplified version of wrapped_as_torch_optimizer_for_ddp.KWNS4/psgd.KronWhiten for single-GPU/DDP training. 
    Important tips:
        initial value lr_preconditioner=0.5 is too high and needs to anneal to ~0.1 (not too small as eps(bf16)~0.01);
        initial value preconditioner_update_probability=1.0 is too high and needs to anneal to 0.01~0.1;
        when loaded from a ckpt with fp32 param, the preconditioner may be upcasted to fp32 and need to restore to bf16.    
    """
    def __init__(
            self,
            params,
            preconditioner_max_size=float("inf"), 
            preconditioner_max_skew=1.0, # for >=2D tensor: 0.0 => all diagonal Q; inf => all dense Q
            preconditioner_init_scale=1.0, # P0 = preconditioner_init_scale^2 * I; set to smaller values if unsure
            lr=3e-4, # the lr_params in PSGD; PSGD has two lrs 
            lr_preconditioner=0.5, # Quickly anneal down to ~ 0.1; don't anneal to ~ 0.01 as eps(bf16) ~ 0.01    
            betaL=0.9, # larger/smaller betaL => longer/shorter history of momentums for whitening
            damping=1e-9, # roughly the eps in Adam(W) 
            momentum=0.9, # roughly the beta1 in Adam(W)
            nesterov=False,
            weight_decay=0.0, 
            decoupled_weight_decay=True, # True for decoupled weight decay; False for the classic weight decay  
            grad_clip_max_amps=(2.0, 10.0), # clip grad with thresholds (max average amplitude, max element-wise amplitude) 
            preconditioner_update_probability=1.0, # Quickly anneal to 0.01 ~ 0.1 to save computations
            resync_every=1_000_000, # resync every # steps if nondeterministic matmul diverges states too much; generally no need.   
    ):
        assert preconditioner_max_size >= 0.0
        assert preconditioner_max_skew >= 0.0
        assert preconditioner_init_scale > 0.0
        assert lr > 0.0
        assert 0.0 < lr_preconditioner < 1.0
        assert 0.0 <= betaL <= 1.0
        assert damping >= 0.0
        assert 0.0 <= momentum <= 1.0
        assert isinstance(nesterov, bool)
        assert weight_decay >= 0.0
        assert isinstance(decoupled_weight_decay, bool)
        assert grad_clip_max_amps[1] >= grad_clip_max_amps[0] >= 1.0 
        assert 0.0 <= preconditioner_update_probability <= 1.0
        assert resync_every > 0

        defaults = {
            "preconditioner_max_size": preconditioner_max_size, 
            "preconditioner_max_skew": preconditioner_max_skew,
            "preconditioner_init_scale": preconditioner_init_scale,
            "lr": lr,
            "lr_preconditioner": lr_preconditioner, 
            "betaL": betaL, 
            "damping": damping, 
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "grad_clip_max_amps": grad_clip_max_amps, 
            "preconditioner_update_probability": preconditioner_update_probability,
        }
        super().__init__(params, defaults)

        self._step = 0
        self._resync_every = resync_every

        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed: # if True, assume multi-GPU DDP training; important to sync the rng states
            # state = torch.get_rng_state().cuda() # assume nccl backend
            # torch.distributed.broadcast(state, src=0)
            # self.cpu_rng_state = state.cpu()
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
            # external_cpu_rng_state = torch.get_rng_state()
            # torch.set_rng_state(self.cpu_rng_state)
            external_cuda_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.cuda_rng_state)

        resync_state = self.is_distributed and ((self._step + 1) % self._resync_every == 0) # resync state flag; False at step=0
        for group in self.param_groups:
            momentum = group["momentum"]
            max_avg_amp, max_element_amp = group["grad_clip_max_amps"]
            prb = group["preconditioner_update_probability"]
            update_P = int(self._step * prb + 1) > int((self._step - 1) * prb + 1) # +1 so that update_P=True for step=0
            balance_Q = int(self._step * prb * 0.01) > int((self._step - 1) * prb * 0.01) # balance Q every 100 updates; False at step=0 
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                wd = group["weight_decay"]
                if wd > 0.0: 
                    if group["decoupled_weight_decay"]:
                        p.mul_(1.0 - wd * group["lr"])
                    else:
                        grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0: # initialization
                    grad = _tensorize(grad)
                    state["ema"] = torch.zeros_like(grad)
                    state["QL"] = init_kron(grad.to(torch.bfloat16), 
                                            Scale=group["preconditioner_init_scale"], 
                                            max_size=group["preconditioner_max_size"], 
                                            max_skew=group["preconditioner_max_skew"])
                    state["step"] = 0
                else:
                    grad = grad.reshape_as(state["ema"]) # memory_format=torch.channels_last Conv could break view_as

                update_fn, apply_fn = _dispatch(state["QL"][0])

                t = state["step"]
                beta = min(t/(t + 1), momentum)
                state["ema"].lerp_(grad, 1.0 - beta) # state["ema"].mul_(beta).add_(grad, alpha=1.0 - beta) 
                state["step"] += 1

                if group["nesterov"]: # transfer fn propto: m/(1 - m*z^{-1}) + 1
                    update = grad.lerp(state["ema"], beta).to(torch.bfloat16) # beta * state["ema"] + (1.0 - beta) * grad 
                else: # transfer fn propto: 1/(1 - m*z^{-1}); less high frequency 
                    update = state["ema"].to(torch.bfloat16)

                if update_P:
                    update_fn(state["QL"], update, 
                              lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])

                h = apply_fn(state["QL"], update)

                avg_amp = torch.linalg.vector_norm(h) * h.numel() ** -0.5 # torch.sqrt(torch.mean(h * h))
                h *= torch.clamp(max_avg_amp/avg_amp, max=1.0) # ok with avg_amp = 0.0
                h.clamp_(min=-max_element_amp, max=max_element_amp) 
                p.sub_(h.view_as(p), alpha=group["lr"])

                # balance Q (optional); resync state occasionally if matmul is not deterministic (generally no need) 
                # no need to compile/capture this part if you use torch.compile/cuda_graph 
                if balance_Q:
                    Q = state["QL"][0]
                    if len(Q) > 1:
                        _balance_n(Q)
                if resync_state:
                    torch.distributed.broadcast(p, src=0)
                    torch.distributed.broadcast(state["ema"], src=0)
                    for q, ell in zip(*state["QL"]):
                        torch.distributed.broadcast(q, src=0)
                        torch.distributed.broadcast(ell, src=0)

        if self.is_distributed: # save internal rng states; recover external rng states 
            # self.cpu_rng_state = torch.get_rng_state()
            # torch.set_rng_state(external_cpu_rng_state)
            self.cuda_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(external_cuda_rng_state)

        self._step += 1
        return loss

