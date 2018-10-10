import torch
from torch.autograd import grad

import preconditioned_stochastic_gradient_descent as psgd 
from mnist_autoencoder_data_model_loss import get_batches, Ws, train_criterion, test_criterion, device

# initialize preconditioners with constant*(identity matrices)
Qs = [[0.1*torch.eye(W.shape[0]).to(device), torch.eye(W.shape[1]).to(device)] for W in Ws]
step_size = 0.1#normalized step size
grad_norm_clip_thr = 100.0#the size of trust region
# begin iteration here
TrainLoss = []
TestLoss = []
echo_every = 100#iterations
for num_iter in range(5000):
    x, y = get_batches( )    
    # calculate the loss and gradient
    loss = train_criterion(Ws, x, y)
    grads = grad(loss, Ws, create_graph=True)
    TrainLoss.append(loss.item())
    
    v = [torch.randn(W.shape).to(device) for W in Ws]
    #grad_v = sum([torch.sum(g*d) for (g, d) in zip(grads, v)])
    #hess_v = grad(grad_v, Ws)
    Hv = grad(grads, Ws, v)#replace the above two lines, due to Bulatov
    #Hv = grads # just let Hv=grads if you use Fisher type preconditioner        
    with torch.no_grad():
        Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= step_adjust*step_size*pre_grads[i]
            
        if (num_iter+1) % echo_every == 0:
            loss = test_criterion(Ws)
            TestLoss.append(loss.item())
            print('train loss: {}; test loss: {}'.format(TrainLoss[-1], TestLoss[-1]))