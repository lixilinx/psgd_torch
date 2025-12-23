import torch
import psgd

class WhitenMomentumNS4(torch.optim.Optimizer):
    """
    Whiten momentum with online Newton-Schulz iteration for inverse 4th root of momentum correlation matrix. 
    Largely corresponds to psgd.KronWhiten with dQ="Q0.5EQ1.5", whiten_grad=False and only real param optimization. 
    Wrapped for distributed training, say FSDP2, that uses Pytorch DTensor. 
    This wrapping preconditions each slice of a DTensor gradient independently, not an optimal but acceptable solution.  
    """

    def __init__(
            self,
            params,
            preconditioner_max_size=float("inf"), 
            preconditioner_max_skew=1.0, # 0.0 => all diagonal Q; inf => all dense Q
            preconditioner_init_scale=1.0, # P0 = preconditioner_init_scale^2 * I; set to small value for warmup
            lr_params=3e-4, # roughly sqrt((1-momentum)/(1+momentum)) * 1e-3   
            lr_preconditioner=0.5, # don't anneal to < 0.1 for bfloat16 preconditioner as eps(bf16) ~ 0.01    
            betaL=0.9, 
            damping=1e-9, # roughly the eps=1e-8 in Adam 
            momentum=0.9, # if momentum is not helpful, reduce or zero it and increase lr_params
            weight_decay=0.0, # L2 regularization, not decoupled wd 
            grad_clip_max_amps=(2.0, 10.0), # clip grad with thresholds (max average amplitude, max element-wise amplitude) 
            preconditioner_update_probability=1.0, # quickly anneal to 0.01 ~ 0.1 to save computations 
            preconditioner_dtype:torch.dtype|None=torch.bfloat16, # bf16 should be good enough for most problems
            update_preconditioner_first=True, # True for biased updates; False for unbiased updates. 
            resync_every=1000_000, # resync every # steps if states divergence too much due to nondeterministic matmul; NOT implemented and generally no need   
    ):
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
            "grad_clip_max_amps": grad_clip_max_amps, 
            "preconditioner_update_probability": preconditioner_update_probability,
            "preconditioner_dtype": preconditioner_dtype,
            "update_preconditioner_first": update_preconditioner_first,
            "resync_every": resync_every,
        }
        super().__init__(params, defaults)

        self.dQ = "Q0.5EQ1.5" # change these 3 lines to select the preconditioner 
        self.update_precond = psgd.update_precond_kron_whiten_q0p5eq1p5
        self.precond_grad = psgd.precond_grad_kron

        # sync rng states 
        state = torch.get_rng_state().cuda() # assume nccl backend
        torch.distributed.broadcast(state, src=0)
        self.cpu_rng_state = state.cpu()

        state = torch.cuda.get_rng_state().cuda()
        torch.distributed.broadcast(state, src=0)
        self.cuda_rng_state = state.cpu()

    @torch.no_grad()
    def step(self):
        external_cpu_rng_state = torch.get_rng_state()
        external_cuda_rng_state = torch.cuda.get_rng_state()
        torch.set_rng_state(self.cpu_rng_state)
        torch.cuda.set_rng_state(self.cuda_rng_state)

        for group in self.param_groups:
            momentum = group["momentum"]
            max_avg_amp, max_element_amp = group["grad_clip_max_amps"]
            if torch.rand([]) < group["preconditioner_update_probability"]:
                update_preconditioner_first, update_preconditioner_last = group["update_preconditioner_first"], not group["update_preconditioner_first"]
            else:
                update_preconditioner_first, update_preconditioner_last = False, False
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                wd = group["weight_decay"]
                if wd > 0.0: # here is the classic wd; just p.mul_(1 - wd * lr_params) for decoupled wd
                    grad = grad.add(p, alpha=wd)

                grad = grad.to_local()
                if grad.numel() == 0: # sharded local grad can be empty 
                    continue 

                preconditioner_dtype = group["preconditioner_dtype"]
                if preconditioner_dtype:
                    grad = grad.squeeze().to(preconditioner_dtype) # squeeze out dim=1; also good to merge small dims if possible
                else:
                    grad = grad.squeeze()

                state = self.state[p]
                if len(state) == 0: # initialization
                    state["QL"], state["exprs"] = psgd.init_kron(grad, 
                                                                 Scale=group["preconditioner_init_scale"], 
                                                                 max_size=group["preconditioner_max_size"], 
                                                                 max_skew=group["preconditioner_max_skew"], 
                                                                 dQ=self.dQ)
                    state["step"] = 0
                    if momentum > 0.0:
                        state["ema"] = torch.zeros_like(grad) # exp moving avg of grad as momentum 

                t = state["step"]
                if momentum > 0.0:
                    beta = min(t/(t + 1), momentum)
                    g = state["ema"]
                    g.mul_(beta).add_(grad, alpha=1 - beta)
                else:
                    g = grad
                state["step"] += 1

                if update_preconditioner_first: # update P before applying on g; biased 
                    self.update_precond(state["QL"], state["exprs"], g, 
                                        lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])
                    
                h = self.precond_grad(state["QL"], state["exprs"], g) # preconditioned momentum 
                avg_amp = torch.sqrt(torch.mean(h*h))
                if avg_amp > max_avg_amp:
                    h *= max_avg_amp/avg_amp
                h.clamp_(min=-max_element_amp, max=max_element_amp) 
                local_p = p.to_local()
                local_p.subtract_(h.view_as(local_p), alpha=group["lr_params"])

                if update_preconditioner_last: # update P after applying on g; unbiased 
                    self.update_precond(state["QL"], state["exprs"], g, 
                                        lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])

                # resync states occasionally if matmul is not deterministic and divergence of replicated state is large 
                if state["step"] % group["resync_every"] == 0:
                    pass # check p.device_mesh and p.placements for subgroups with duplicated states and resync them as in DDP wrapping     

        self.cpu_rng_state = torch.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state()
        torch.set_rng_state(external_cpu_rng_state)
        torch.cuda.set_rng_state(external_cuda_rng_state)


if __name__ == "__main__":
    # toy demo for verification  
    import os
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard as FSDP

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0")) 
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl") 

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.w = torch.nn.Parameter(torch.empty(4, 2))

        def forward(self, x):
            y = (self.w - torch.rand_like(self.w)) * x
            return torch.mean(y * y)

    # test with 2 x 2 = 4 GPUs
    mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard"))

    with torch.device("meta"):
        model = ToyModel()
    model.to_empty(device="cuda")
    FSDP(model, mesh=mesh_2d)
    model.w.data *= 0
    opt = WhitenMomentumNS4(model.parameters())
    for _ in range(1000):
        opt.zero_grad()
        loss = model(torch.randn_like(model.w.full_tensor()))
        loss.backward()
        opt.step()

    # let's verify the wrapping.
    # pairs with the same local weights and preconditioner: (rank 0, rank 2) and (rank 1, rank 3). 
    # pairs with different local weights and preconditioner: (rank 0, rank 1) and (rank 2, rank 3).
    # full weights should be the same for all ranks.    
    print(f"rank {rank} got loss {loss}") # different GPU may have different loss 
    print(f"rank {rank} got local model weights {model.w.to_local()}") 
    print(f"rank {rank} got full model weights {model.w.full_tensor()}") 

    torch.distributed.destroy_process_group()
    