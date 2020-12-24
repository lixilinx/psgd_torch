"""LSTM network with the classic delayed XOR problem. Common but hard to learn the XOR relation between two events with lag
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import preconditioned_stochastic_gradient_descent as psgd 

batch_size, seq_len = 128, 50 # increasing sequence_length
dim_in, dim_hidden, dim_out = 2, 30, 1 # or decreasing dimension_hidden_layer will make learning harder

def generate_train_data( ):
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i,:,0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand()*0.1*seq_len))
        i2 = int(np.floor(np.random.rand()*0.4*seq_len + 0.1*seq_len))             
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i,i1,0] == x[i,i2,0]: # XOR
            y[i] = -1.0 # lable 0
        else:
            y[i] = 1.0  # lable 1
            
    #tranpose x to format (sequence_length, batch_size, dimension_of_input)  
    return torch.tensor(np.transpose(x, [1, 0, 2])), torch.tensor(y)

class LSTM_net(torch.nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        W1 = 0.1*torch.randn(dim_in + 2*dim_hidden + 1, 4*dim_hidden)
        W1[-1, dim_hidden:2*dim_hidden] += 1.0 # forget gate with large bias to encourage long term memory 
        W1[:, 2*dim_hidden:3*dim_hidden] *= 2.0 # cause tanh(x)=2*sigmoid(2*x) - 1
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(0.1*torch.randn(dim_hidden + 1, dim_out))
        
    def forward(self, xs):
        h, c = torch.zeros(batch_size, dim_hidden), torch.zeros(batch_size, dim_hidden) # initial hidden and cell states
        for x in torch.unbind(xs):
            # the same as https://github.com/lixilinx/psgd_tf/blob/master/lstm_with_xor_problem.py, slightly twisted for speed
            ifgo = torch.cat([x, h, c], dim=1) @ self.W1[:-1] + self.W1[-1] # here cell state is in the input feature as well
            i, f, g, o = torch.chunk(torch.sigmoid(ifgo), 4, dim=1)
            c = f*c + i*(2.0*g - 1.0) # new cell state; note that tanh(x)=2*sigmoid(2*x) - 1
            h = o*torch.tanh(c) # new hidden state
        return h @ self.W2[:-1] + self.W2[-1]
        
lstm_net = LSTM_net()

@torch.jit.script
def train_loss(xy_pair): # logistic loss
    # type: (Tuple[Tensor, Tensor]) -> Tensor
    return -torch.mean(torch.log(torch.sigmoid( xy_pair[1]*lstm_net(xy_pair[0]) )))

Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in lstm_net.parameters()]
lr = 0.02
grad_norm_clip_thr = 1.0
Losses = []
for num_iter in range(100000):
    loss = train_loss(generate_train_data())
    grads = torch.autograd.grad(loss, lstm_net.parameters(), create_graph=True)
    vs = [torch.randn_like(W) for W in lstm_net.parameters()]
    Hvs = torch.autograd.grad(grads, lstm_net.parameters(), vs) 
    with torch.no_grad():
        Qs = [psgd.update_precond_kron(Qlr[0], Qlr[1], v, Hv) for (Qlr, v, Hv) in zip(Qs, vs, Hvs)]
        pre_grads = [psgd.precond_grad_kron(Qlr[0], Qlr[1], g) for (Qlr, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        lr_adjust = min(grad_norm_clip_thr/grad_norm, 1.0)
        [W.subtract_(lr_adjust*lr*g) for (W, g) in zip(lstm_net.parameters(), pre_grads)]                
    Losses.append(loss.item())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1])) 
    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)