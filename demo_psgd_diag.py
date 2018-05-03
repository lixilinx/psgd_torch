import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

import preconditioned_stochastic_gradient_descent as psgd 
from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# initialize preconditioners with identity matrices
Qs = [torch.ones(W.size()) for W in Ws]
# begin iteration here
step_size = 0.02
grad_norm_clip_thr = 1.0
Loss = []
for num_iter in range(10000):
    x, y = get_batches( )
    
    # calculate loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws, create_graph=True)
    Loss.append(loss.item())
    
    # update preconditioners
    delta = [torch.randn(W.size()) for W in Ws]
    grad_delta = sum([torch.sum(g*d) for (g, d) in zip(grads, delta)])
    hess_delta = grad(grad_delta, Ws)
    with torch.no_grad():
        Qs = [psgd.update_precond_diag(q, dw, dg) for (q, dw, dg) in zip(Qs, delta, hess_delta)]        
        # update Ws
        pre_grads = [psgd.precond_grad_diag(q, g) for (q, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= step_adjust*step_size*pre_grads[i]
            
        if num_iter % 100 == 0:
            print('training loss: {}'.format(Loss[-1]))
    
plt.semilogy(Loss)
