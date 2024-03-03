import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")
import preconditioned_stochastic_gradient_descent as psgd

#torch.set_default_device(torch.device("cpu"))
#torch.set_default_device(torch.device("cuda"))

"""
first let's check that our guess of the usage of function
    torch._VF.rnn_tanh()
is correct since there is no documument to follow.
"""
batch = 1
length = 1
input_size = 3
hidden_size = 5
num_layers = 2

inputs = torch.randn(batch, length, input_size)  # (Batch, Length, Feature) format
hx = torch.randn(num_layers, batch, hidden_size)

# affine params, row major format 
param_1st_layer = torch.randn(input_size + hidden_size + 1, hidden_size)
param_2nd_layer = torch.randn(2 * hidden_size + 1, hidden_size)

# prepare the param list for _VF.rnn_tanh to consume
weights = [
    param_1st_layer[: -hidden_size - 1].t().contiguous(), # this is annoying as torch RNN needs an extra transpose of weights 
    param_1st_layer[-hidden_size - 1 : -1].t().contiguous(),
    param_1st_layer[-1],
    torch.zeros_like(param_1st_layer[-1]),
    param_2nd_layer[: -hidden_size - 1].t().contiguous(),
    param_2nd_layer[-hidden_size - 1 : -1].t().contiguous(),
    param_2nd_layer[-1],
    torch.zeros_like(param_2nd_layer[-1]),
]

# get _VF.rnn_tanh's result
results = torch._VF.rnn_tanh(
    inputs,
    hx,
    weights,
    has_biases=True,
    num_layers=num_layers,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=True,
)

# get my own results
out_1st_layer = torch.tanh(
    inputs[0] @ weights[0].t() + hx[0] @ weights[1].t() + weights[2] + weights[3]
)
out_2nd_layer = torch.tanh(
    out_1st_layer @ weights[4].t() + hx[1] @ weights[5].t() + weights[6] + weights[7]
)
print(f"the error should be close to 0: {out_2nd_layer - results[0]}")


"""
Now wrap our own RNN
"""
class AffineRNN(torch.nn.Module):
    """
    Class AffineRNN wraps function
        torch._VF.rnn_tanh 
    into a class.
    I only consider the case with:
        has_biases=True, bidirectional=False, batch_first=True, nonl=tanh
        
    The affine transform of each layer is 
    
                                        [    w_ih,  ]
       [inputs, hidden states, 1]   @   [    w_hh,  ]
                                        [    bias,  ]
                                        
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(AffineRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        def rnn_weight_initialization(input_size, hidden_size):
            """
            the weight matrix is store as
            [   w_ih,
                w_hh,
                bias, ]
            
            The normal_ initialization might be better than uniform_ (keep the same variance)? 
            """
            w = torch.empty(input_size + hidden_size, hidden_size).normal_(
                std=(3*(input_size + hidden_size)) ** (-0.5)
            )
            # w = torch.empty(input_size + hidden_size, hidden_size).uniform_(
            #     -(input_size+hidden_size)**(-0.5), (input_size+hidden_size)**(-0.5)
            # )
            w[input_size:] = torch.linalg.qr(w[input_size:])[0]
            b = torch.zeros(1, hidden_size)
            return torch.cat([w, b])

        self.param0 = torch.nn.Parameter(rnn_weight_initialization(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.register_parameter(
                f"param{i+1}",
                torch.nn.Parameter(rnn_weight_initialization(hidden_size, hidden_size)),
            )

    def forward(self, inputs, hx):
        # prepare weights for _VF.rnn_tanh
        weights = []
        for i in range(self.num_layers):
            p = self.get_parameter(f"param{i}")
            weights.extend(
                [
                    p[: -self.hidden_size - 1].t().contiguous(), # ok, we can save this .t().contiguous() if everyone follows row major convention 
                    p[-self.hidden_size - 1 : -1].t().contiguous(),
                    p[-1],
                    torch.zeros_like(p[-1]),
                ]
            )
        # call _VF.rnn_tanh to return the (outputs, hidden states)
        return torch._VF.rnn_tanh(
            inputs,
            hx,
            weights,
            has_biases=True,
            num_layers=self.num_layers,
            dropout=self.dropout,
            train=self.training, # this self.training is inherited from torch.nn.Module 
            bidirectional=False,
            batch_first=True,
        )

# let's test our wrapped RNN
batch, length, input_size, hidden_size, num_layers = 4, 6, 3, 5, 2
rnn = AffineRNN(input_size, hidden_size, num_layers=num_layers)
inputs = torch.randn(batch, length, input_size)  
hx = torch.randn(num_layers, batch, hidden_size)
results = rnn(inputs, hx)

"""
Lastly, let's solve the delayed XOR problem with the Affine RNN and preconditioner
"""
device = hx.device
batch_size, seq_len = (128, 16,) 
dim_in, dim_hidden, dim_out = (2, 30, 1,)

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

    return [torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)]

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = AffineRNN(dim_in, dim_hidden)
        self.linear_wb = torch.nn.Parameter(torch.cat([0.1 * torch.randn(dim_hidden, dim_out), 
                                                       torch.zeros(1, dim_out)]))

    def forward(self, xs):
        _, h = self.rnn(xs, torch.zeros(1, batch_size, dim_hidden))
        return h[0] @ self.linear_wb[:-1] + self.linear_wb[-1]

model = Model()

opt = psgd.Affine(
    model.parameters(), preconditioner_init_scale=None, lr_params=0.01, lr_preconditioner=0.01, 
    grad_clip_max_norm=1.0, exact_hessian_vector_product=False
)
# opt = psgd.LRA(
#     model.parameters(), preconditioner_init_scale=None, lr_params=0.01, lr_preconditioner=0.01, 
#     grad_clip_max_norm=1.0, exact_hessian_vector_product=False
# )

def train_loss(xy_pair):  # logistic loss
    return -torch.mean(torch.log(torch.sigmoid(xy_pair[1] * model(xy_pair[0]))))

Losses = []
for num_iter in range(100000):
    train_data = generate_train_data()

    def closure():
        return train_loss(train_data) 

    Losses.append(opt.step(closure).item())
    print(f"Iteration: {num_iter + 1}; loss: {Losses[-1]}")

    if Losses[-1] < 0.1:
        print("Deemed to be successful and ends training")
        break
plt.plot(Losses)