import torch
import psgd

class KWNS4(torch.optim.Optimizer):
    """
    Kronecker-product whitening preconditioner fitted with online Newton-Schulz iteration for inverse 4th root of gradient/momentum correlation matrix. 
    Largely corresponds to psgd.KronWhiten with dQ="Q0.5EQ1.5" and only real param optimization. 
    Wrapped for single-GPU and multi-GPU DDP training. 

    The preconditioner can be fitted with two choices:  
        1st): P is fitted by whitening gradient; then apply P on the momentum for param update. 
        2nd): P is fitted by whitening momentum; the whitened momentum is used for param update.
    For the 2nd choice, we need to reduce lr_params accordingly with increased momentum, roughly by:
        (a base lr_params) * sqrt((1 - momentum)/(1 + momentum)) 
    where the base lr_params can be 1e-3 following Adam(W).  

    To migrate AdamW decoupled weight decay setting to PSGD, one can let 
        (PSGD lr_params) * (PSGD weight_decay) = (AdamW lr) * (AdamW weight_decay)
    For the L2 regularization, one can keep similar weight decays across different optimizers.  

    For more details, please check the inline comments for hyper-param settings.
    Don't forget to quickly anneal down lr_preconditioner and preconditioner_update_probability. 
    """

    def __init__(
            self,
            params,
            whiten_grad=False, # True for whitening gradient; False for whitening momentum 
            preconditioner_max_size=float("inf"), 
            preconditioner_max_skew=1.0, # 0.0 => all diagonal Q; inf => all dense Q
            preconditioner_init_scale=1.0, # P0 = preconditioner_init_scale^2 * I; set to smaller values if unsure
            lr_params=2e-4, # following Adam(W), 1e-3 for whiten_grad=True; sqrt((1-momentum)/(1+momentum)) * 1e-3 for whiten_grad=False   
            lr_preconditioner=0.5, # Quickly anneal down to 0.01 ~ 0.1. Don't anneal to < 0.1 for bfloat16 preconditioner as eps(bf16) ~ 0.01    
            betaL=0.9, 
            damping=1e-9, # roughly the eps in Adam(W) 
            momentum=0.9, # roughly the beta1 in Adam(W)
            weight_decay=0.05, # following AdamW, we let lr_params * weight_decay = 2e-4 * 0.05 = (AdamW lr 1e-3) * (AdamW wd 0.01)
            decoupled_weight_decay=True, # True for decoupled weight decay; False for the classic weight decay  
            grad_clip_max_amps=(2.0, 10.0), # clip grad with thresholds (max average amplitude, max element-wise amplitude) 
            preconditioner_update_probability=1.0, # Quickly anneal to 0.01 ~ 0.1 to save computations
            preconditioner_dtype:torch.dtype|None=torch.bfloat16, # bf16 should be good enough for most problems
            update_preconditioner_first=True, # True for biased updates; False for unbiased updates. 
            resync_every=1000_000, # resync every # steps if nondeterministic matmul diverges states too much; generally no need.   
    ):
        assert whiten_grad in (False, True)
        assert preconditioner_max_size >= 0.0
        assert preconditioner_max_skew >= 0.0
        assert preconditioner_init_scale > 0.0
        assert lr_params > 0.0
        assert 0.0 < lr_preconditioner < 1.0
        assert 0.0 <= betaL <= 1.0
        assert damping >= 0.0
        assert 0.0 <= momentum < 1.0
        assert weight_decay >= 0.0
        assert decoupled_weight_decay in (False, True)
        assert grad_clip_max_amps[1] >= grad_clip_max_amps[0] >= 1.0 
        assert 0.0 < preconditioner_update_probability <= 1.0
        assert preconditioner_dtype in (None, torch.bfloat16, torch.float32)
        assert update_preconditioner_first in (False, True)
        assert resync_every > 0
        if not whiten_grad:
            assert momentum > 0.0, "Cannot whiten momentum if momentum setting is zero."

        defaults = {
            "whiten_grad": whiten_grad,
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
            "preconditioner_dtype": preconditioner_dtype,
            "update_preconditioner_first": update_preconditioner_first,
            "resync_every": resync_every,
        }
        super().__init__(params, defaults)

        self.dQ = "Q0.5EQ1.5" # one can change these 3 lines to switch the preconditioner 
        self.update_precond = psgd.update_precond_kron_whiten_q0p5eq1p5
        self.precond_grad = psgd.precond_grad_kron

        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed: # if True, assume multi-GPU DDP training; important to sync the rng states
            state = torch.get_rng_state().cuda() # assume nccl backend
            torch.distributed.broadcast(state, src=0)
            self.cpu_rng_state = state.cpu()

            state = torch.cuda.get_rng_state().cuda() # assume nccl backend 
            torch.distributed.broadcast(state, src=0)
            self.cuda_rng_state = state.cpu()

    @torch.no_grad()
    def step(self):
        if self.is_distributed: # sync internal rng states; save external rng states 
            external_cpu_rng_state = torch.get_rng_state()
            external_cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(self.cpu_rng_state)
            torch.cuda.set_rng_state(self.cuda_rng_state)

        for group in self.param_groups:
            momentum = group["momentum"]
            max_avg_amp, max_element_amp = group["grad_clip_max_amps"]
            updateP_first, updateP_last = ((group["update_preconditioner_first"], not group["update_preconditioner_first"]) 
                                           if torch.rand([]) < group["preconditioner_update_probability"] else (False, False))
                
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

                grad = grad.squeeze() # squeeze out singleton dims; also good to merge small dims here 
                preconditioner_dtype = group["preconditioner_dtype"]
                if preconditioner_dtype:
                    grad = grad.to(preconditioner_dtype) 

                state = self.state[p]
                if len(state) == 0: # initialization
                    state["QL"], state["exprs"] = psgd.init_kron(grad, 
                                                                 Scale=group["preconditioner_init_scale"], 
                                                                 max_size=group["preconditioner_max_size"], 
                                                                 max_skew=group["preconditioner_max_skew"], 
                                                                 dQ=self.dQ)
                    state["step"] = 0
                    state["ema"] = None if momentum == 0.0 else torch.zeros_like(grad) # exp moving avg of grad as momentum 

                t = state["step"]
                if momentum > 0.0:
                    beta = min(t/(t + 1), momentum)
                    state["ema"].mul_(beta).add_(grad, alpha=1.0 - beta)
                state["step"] += 1

                to_be_whitened = grad if group["whiten_grad"] else state["ema"]
                if updateP_first: # update P before applying on grad/momentum; biased 
                    self.update_precond(state["QL"], state["exprs"], to_be_whitened, 
                                        lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])

                to_be_preconded = grad if momentum == 0.0 else state["ema"]    
                h = self.precond_grad(state["QL"], state["exprs"], to_be_preconded)

                avg_amp = torch.sqrt(torch.mean(h*h))
                if avg_amp > max_avg_amp:
                    h *= max_avg_amp/avg_amp
                h.clamp_(min=-max_element_amp, max=max_element_amp) 
                p.subtract_(h.view_as(p), alpha=group["lr_params"])

                if updateP_last: # update P after applying on grad/momentum; unbiased 
                    self.update_precond(state["QL"], state["exprs"], to_be_whitened, 
                                        lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])

                # resync states occasionally if matmul is not deterministic and state divergence is large 
                if self.is_distributed and (state["step"] % group["resync_every"] == 0):
                    torch.distributed.broadcast(p, src=0)
                    if momentum > 0.0:
                        torch.distributed.broadcast(state["ema"], src=0)
                    for q, ell in zip(*state["QL"]):
                        torch.distributed.broadcast(q, src=0)
                        torch.distributed.broadcast(ell, src=0)

        if self.is_distributed: # save internal rng states; recover external rng states 
            self.cpu_rng_state = torch.get_rng_state()
            self.cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(external_cpu_rng_state)
            torch.cuda.set_rng_state(external_cuda_rng_state)


if __name__ == "__main__":
    # toy demo for verification  
    import os
    from torch.nn.parallel import DistributedDataParallel as DDP

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0")) 
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")  

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.w = torch.nn.Parameter(torch.randn(1, 2, 3, 4))

        def forward(self, x):
            y = (self.w - torch.rand_like(self.w)) * x
            return torch.mean(y * y)

    model = DDP(ToyModel().to(device), device_ids=[local_rank])
    opt = KWNS4(model.parameters())
    for _ in range(1000):
        opt.zero_grad()
        loss = model(torch.randn_like(model.module.w))
        loss.backward()
        opt.step()

    # let's verify the wrapping  
    print(f"rank {rank} got loss {loss}") # different GPU may have different loss 
    print(f"rank {rank} got model weights {model.module.w}") # model weights should be close (no bitwise match guarantee in general)
    print(f"rank {rank} got preconditioner states {opt.state[model.module.w]['QL']}") # close but no bitwise match guarantee

    torch.distributed.destroy_process_group()
    
