"""RNN network with the classic delayed XOR problem.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import preconditioned_stochastic_gradient_descent as psgd 

device = torch.device('cpu')
batch_size, seq_len = 128, 20          # increasing sequence_length
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
    return [torch.tensor(np.transpose(x, [1, 0, 2])).to(device), 
            torch.tensor(y).to(device)]

# generate a random orthogonal matrix for recurrent matrix initialization 
def get_rand_orth( dim ):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return torch.tensor(q, dtype=torch.float32).to(device)

class RNN_net(torch.nn.Module):
    def __init__(self):
        super(RNN_net, self).__init__()
        self.W1x = torch.nn.Parameter(0.1*torch.randn(dim_in, dim_hidden))
        self.W1h = torch.nn.Parameter(get_rand_orth(dim_hidden))
        self.b1  = torch.nn.Parameter(torch.zeros([]))
        self.W2  = torch.nn.Parameter(0.1*torch.randn(dim_hidden, dim_out))
        self.b2  = torch.nn.Parameter(torch.zeros([]))
        
    def forward(self, xs):
        h = torch.zeros(batch_size, dim_hidden, device=device)
        for x in torch.unbind(xs):
            h = torch.tanh(x @ self.W1x + h @ self.W1h + self.b1)
        return h @ self.W2 + self.b2
        
rnn_net = RNN_net().to(device)

def train_loss(xy_pair): # logistic loss
    return -torch.mean(torch.log(torch.sigmoid( xy_pair[1]*rnn_net(xy_pair[0]) )))

num_paras = sum([torch.numel(W) for W in rnn_net.parameters()])
order_UVd = 10
"""
UVd preconditioner initialization:
    the first part with shape [order_UVd, num_paras] is for U;
    the second part with shape [order_UVd, num_paras] is for V;
    the last part with shape [1, num_paras] is for the diagonal matrix diag(d).
We concat them into one tensor with flag requires_grad=False. 
"""
UVd = torch.cat([torch.randn(2*order_UVd, num_paras)/(order_UVd*num_paras)**0.5,
                 torch.ones(1, num_paras)], dim=0).to(device)
lr = 0.01
grad_norm_clip_thr = 1.0
Losses = []
for num_iter in range(100000):
    loss = train_loss(generate_train_data())
    grads = torch.autograd.grad(loss, rnn_net.parameters(), create_graph=True)
    vs = [torch.randn_like(W) for W in rnn_net.parameters()]
    Hvs = torch.autograd.grad(grads, rnn_net.parameters(), vs) 
    with torch.no_grad():
        UVd = psgd.update_precond_UVd(UVd, vs, Hvs)
        pre_grads = psgd.precond_grad_UVd(UVd, grads)
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        lr_adjust = min(grad_norm_clip_thr/grad_norm, 1.0)
        [W.subtract_(lr_adjust*lr*g) for (W, g) in zip(rnn_net.parameters(), pre_grads)]                
    Losses.append(loss.item())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1])) 
    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)