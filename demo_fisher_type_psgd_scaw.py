"""
Let F = empirical Fisher information matrix. We consider P = F**(-0.5) vs P**2 = F**(-1)

P = F**(-0.5): Adam and RMSProp use it with diagonal form. Their successes support this choice.

P**2 = F**(-1): Amari's natural gradient theory support this choice.

For this toy example, P**2 is better than P. Possible reasons: training data have enough diversity
such that F is well conditioned; assuming prediction error as Gaussian is accurate here. 

For many other tasks involving real word data, P might be better than P**2. Possible reasons: F is 
ill conditioned as training data have low intrinsic dimensions while model is over parameterized; 
mismatch between the assumed and real pdf models for data.    
"""
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

import preconditioned_stochastic_gradient_descent as psgd 
from rnn_add_problem_data_model_loss import get_batches, Ws, train_criterion

# initialize preconditioners with identity matrices
Qs = [[torch.eye(W.shape[0]), torch.ones((1, W.shape[1]))] for W in Ws]

# begin iteration here
step_size = 0.02
damping = 1e-3
grad_norm_clip_thr = 1.0
Loss = []
for num_iter in range(10000):
    x, y = get_batches( )
    
    # calculate loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws)
    Loss.append(loss.item())
    delta = [torch.randn(W.shape) for W in Ws]
    with torch.no_grad():
        Qs = [psgd.update_precond_scaw(q[0], q[1], dw, dg + damping*dw) for (q, dw, dg) in zip(Qs, delta, grads)]
        pre_grads = [psgd.precond_grad_scaw(q[0], q[1], g) for (q, g) in zip(Qs, grads)]# P*g: whitened gradients
        pre_grads = [psgd.precond_grad_scaw(q[0], q[1], g) for (q, g) in zip(Qs, pre_grads)]# P*P*g: natural gradients
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= step_adjust*step_size*pre_grads[i]
            
        if num_iter % 100 == 0:
            print('training loss: {}'.format(Loss[-1]))
    
plt.semilogy(Loss)
