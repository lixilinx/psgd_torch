import os

import psgd

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class WhitenMomentumNS4(torch.optim.Optimizer):
    """
    Whiten momentum with online Newton-Schulz iteration for inverse 4th root of momentum correlation matrix. 
    Largely corresponds to psgd.KronWhiten with dQ="Q0.5EQ1.5" and whiten_grad=False.
    """

    def __init__(
            self,
            params,
            preconditioner_max_size=float("inf"), 
            preconditioner_max_skew=1.0, # 0.0 => all diagonal Q; inf => all dense Q
            preconditioner_init_scale=1.0, # P0 = preconditioner_init_scale^2 * I; smaller is safer
            lr_params=0.0003, # roughly sqrt((1+momentum)/(1-momentum)) times smaller than Adam's lr  
            lr_preconditioner=0.1, 
            betaL=0.9, 
            damping=1e-9, # roughly the eps=1e-8 in Adam 
            momentum=0.9, # momentum is not always helpful!
            weight_decay=0.0, # L2 regularization, not decoupled wd 
            grad_clip_max_amp=2.0, # unusual to have amplitude>>1.0; clip if too big 
            preconditioner_update_probability=0.1, # successive momentum are correlated, no need to update every step 
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
            "grad_clip_max_amp": grad_clip_max_amp, 
            "preconditioner_update_probability": preconditioner_update_probability,
        }
        super().__init__(params, defaults)

        self.dQ = "Q0.5EQ1.5" # change these 3 lines to select the preconditioner 
        self.update_precond = psgd.update_precond_kron_whiten_q0p5eq1p5
        self.precond_grad = psgd.precond_grad_kron

        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed:
            # we assume it's distributed; important to let all the copies have synced rng states
            self.device = self.param_groups[0]["params"][0].device
            state = torch.get_rng_state().to(self.device) # assume nccl backend
            torch.distributed.broadcast(state, src=0)
            self.cpu_rng_state = state.cpu()

            state = torch.cuda.get_rng_state().to(self.device)
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad 

                wd = group["weight_decay"]
                if wd > 0.0:
                    grad = grad.add(p, alpha=wd)

                grad = grad.squeeze() # squeeze out dim=1

                state = self.state[p]
                if len(state) == 0: # intialization
                    state["QL"], state["exprs"] = psgd.init_kron(grad, 
                                                                 Scale=group["preconditioner_init_scale"], 
                                                                 max_size=group["preconditioner_max_size"], 
                                                                 max_skew=group["preconditioner_max_skew"], 
                                                                 dQ=self.dQ)
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(grad) # exp moving avg of grad as momentum 

                t = state["step"]
                beta = min(t/(t + 1), group["momentum"])
                state["ema"].mul_(beta).add_(grad, alpha=1 - beta)
                state["step"] += 1

                if torch.rand([]) < group["preconditioner_update_probability"]:
                    self.update_precond(state["QL"], state["exprs"], state["ema"], 
                                        lr=group["lr_preconditioner"], betaL=group["betaL"], damping=group["damping"])
                    
                h = self.precond_grad(state["QL"], state["exprs"], state["ema"]) # preconditioned momentum 
                amp = torch.sqrt(torch.real(torch.mean(h * h.conj())))
                p.subtract_(group["lr_params"]/torch.clamp(amp/group["grad_clip_max_amp"], min=1.0) * h.view_as(p))

        if self.is_distributed: # save internal rng states; recover external rng states 
            self.cpu_rng_state = torch.get_rng_state()
            self.cuda_rng_state = torch.cuda.get_rng_state()
            torch.set_rng_state(external_cpu_rng_state)
            torch.cuda.set_rng_state(external_cuda_rng_state)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(1,2,3,4))

    def forward(self, x):
        y = (self.w - 1) * x
        return torch.real(torch.mean(y * y.conj()))


def demo():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0")) # rank!=local_rank if #(nodes)>1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl") # use gloo for complex tensors; nccl does not support complex tensors! 

    model = DDP(ToyModel().to(device), device_ids=[local_rank])
    opt = WhitenMomentumNS4(model.parameters(), weight_decay=1e-6)
    for _ in range(1000):
        opt.zero_grad()
        loss = model(torch.randn_like(model.module.w))
        loss.backward()
        opt.step()

    # let's check its correctness
    print(f"rank {rank} got loss {loss}") # different GPU should have different loss 
    print(f"rank {rank} got model weights {model.module.w}") # model weights should be close (no bitwise match guarantee)
    print(f"""rank {rank} got preconditioner states {opt.state[model.module.w]["QL"]}""") # close but no bitwise match guarantee


if __name__ == "__main__":
    demo()
