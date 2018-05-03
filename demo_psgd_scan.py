import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

import preconditioned_stochastic_gradient_descent as psgd 
from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# initialize preconditioners with identity matrices
Qs = [[torch.cat([torch.ones((1, W.shape[0])), torch.zeros((1, W.shape[0]))]),
       torch.ones((1, W.shape[1]))] for W in Ws]
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
    delta = [torch.randn(W.shape) for W in Ws]
    grad_delta = sum([torch.sum(g*d) for (g, d) in zip(grads, delta)])
    hess_delta = grad(grad_delta, Ws)
    with torch.no_grad():
        Qs = [psgd.update_precond_scan(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, delta, hess_delta)]
        pre_grads = [psgd.precond_grad_scan(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= step_adjust*step_size*pre_grads[i]
            
        if num_iter % 100 == 0:
            print('training loss: {}'.format(Loss[-1]))
    
plt.semilogy(Loss)
