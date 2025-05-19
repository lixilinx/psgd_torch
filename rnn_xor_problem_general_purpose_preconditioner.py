"""
RNN network with the classic delayed XOR problem.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import psgd

device = torch.device('cpu')
batch_size, seq_len = 128, 16
dim_in, dim_hidden, dim_out = 2, 30, 1

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
    return [torch.from_numpy(np.transpose(x, [1, 0, 2])).to(device),
            torch.from_numpy(y).to(device)]

# generate a random orthogonal matrix for recurrent matrix initialization
def get_rand_orth(dim):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return torch.from_numpy(q.astype(np.float32)).to(device)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W1x = torch.nn.Parameter(0.1 * torch.randn(dim_in, dim_hidden))
        self.W1h = torch.nn.Parameter(get_rand_orth(dim_hidden))
        self.b1 = torch.nn.Parameter(torch.zeros(dim_hidden))
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden, dim_out))
        self.b2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, xs):
        h = torch.zeros(batch_size, dim_hidden, device=device)
        for x in torch.unbind(xs):
            h = torch.tanh(x @ self.W1x + h @ self.W1h + self.b1)
        return h @ self.W2 + self.b2

model = Model().to(device)
opt = psgd.KronWhiten(model.parameters(), preconditioner_init_scale=1.0, 
                      lr_params=1e-3, lr_preconditioner=0.01, grad_clip_max_amp=1.0)

def train_loss(xy_pair):  # logistic loss
    return -torch.mean(torch.log(torch.sigmoid(xy_pair[1] * model(xy_pair[0]))))

Losses = []
for num_iter in range(100000):
    train_data = generate_train_data()
    Losses.append(opt.step(lambda: train_loss(train_data)).item())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1]))

    if Losses[-1] < 0.1:
        print('Successful')
        break
plt.plot(Losses)
