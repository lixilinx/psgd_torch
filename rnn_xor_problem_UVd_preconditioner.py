"""RNN network with the classic delayed XOR problem.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device('cpu')
batch_size, seq_len = 128, 20           # increasing sequence_length
dim_in, dim_hidden, dim_out = 2, 30, 1  # or decreasing dimension_hidden_layer will make learning harder

def generate_train_data():
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i, :, 0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand() * 0.1 * seq_len))
        i2 = int(np.floor(np.random.rand() * 0.4 * seq_len + 0.1 * seq_len))
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i, i1, 0] == x[i, i2, 0]:  # XOR
            y[i] = -1.0  # lable 0
        else:
            y[i] = 1.0  # lable 1

    # tranpose x to format (sequence_length, batch_size, dimension_of_input)
    return [torch.tensor(np.transpose(x, [1, 0, 2])).to(device),
            torch.tensor(y).to(device)]

# generate a random orthogonal matrix for recurrent matrix initialization
def get_rand_orth(dim):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return torch.tensor(q, dtype=torch.float32).to(device)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W1x = torch.nn.Parameter(0.1 * torch.randn(dim_in, dim_hidden))
        self.W1h = torch.nn.Parameter(get_rand_orth(dim_hidden))
        self.b1 = torch.nn.Parameter(torch.zeros([]))
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden, dim_out))
        self.b2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, xs):
        h = torch.zeros(batch_size, dim_hidden, device=device)
        for x in torch.unbind(xs):
            h = torch.tanh(x @ self.W1x + h @ self.W1h + self.b1)
        return h @ self.W2 + self.b2

model = Model().to(device)
# initialize the PSGD optimizer 
opt = psgd.UVd(model.parameters(),
               rank_of_modification=10, preconditioner_init_scale=1.0,
               lr_params=0.01, lr_preconditioner=0.01,
               grad_clip_max_norm=1.0, preconditioner_update_probability=1.0,
               exact_hessian_vector_product=True)

def train_loss(xy_pair):  # logistic loss
    return -torch.mean(torch.log(torch.sigmoid(xy_pair[1] * model(xy_pair[0]))))

Losses = []
for num_iter in range(100000):
    train_data = generate_train_data()

    # rng_state = torch.get_rng_state()
    # rng_cuda_state = torch.cuda.get_rng_state()
    def closure(): 
        # If exact_hessian_vector_product=False and rng is used inside closure, 
        # make sure rng starts from the same state; otherwise, doesn't matter. 
        # torch.set_rng_state(rng_state)
        # torch.cuda.set_rng_state(rng_cuda_state)
        return train_loss(train_data) # return a loss
        # return [train_loss(train_data),] # or return a list with the 1st one being loss

    loss = opt.step(closure)
    Losses.append(loss.item())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1]))
    if num_iter+1 == 1000: # feel free to reschedule these mutable settings
        # opt.lr_params = 0.01
        # opt.lr_preconditioner = 0.01
        # opt.grad_clip_max_norm = 1.0
        # opt.preconditioner_update_probability = 1.0
        opt.exact_hessian_vector_product = False

    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)