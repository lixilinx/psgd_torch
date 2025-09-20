"""
A tiny GPT2 demo. 
The GPT2 model is from
    https://github.com/karpathy/nanoGPT/blob/master/model.py
One can train it on a laptop GPU with 8 GB RAM. 
"""

import copy

import math

import os
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tiktoken

import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")
import psgd

device = torch.device("cuda")

torch.set_default_dtype(torch.bfloat16)

def set_seed(seed):
    # from chatgpt 
    np.random.seed(seed)                   # NumPy RNG
    torch.manual_seed(seed)                # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)           # PyTorch GPU RNG (if using CUDA)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

"""
Prepare the dataset
"""
if not os.path.exists("./wikitext-103-raw/tokenized_data.torch"):
    print(
        """
          Download wikitext from,
              https://wikitext.smerity.com/wikitext-103-raw-v1.zip,
          Unzip it, 
          and copy to local folder
              ./wikitext-103-raw/
          """
    )

    with open("./wikitext-103-raw/wiki.train.raw", encoding="utf8") as data_file:
        raw_train_data = data_file.read()

    with open("./wikitext-103-raw/wiki.valid.raw", encoding="utf8") as data_file:
        raw_eval_data = data_file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    raw_tokenized_train = tokenizer.encode_ordinary(raw_train_data)
    raw_tokenized_eval = tokenizer.encode_ordinary(raw_eval_data)

    train_tokenized = torch.tensor(raw_tokenized_train, dtype=torch.int32)
    eval_tokenized = torch.tensor(raw_tokenized_eval, dtype=torch.int32)

    tokenized_data = {"train": train_tokenized, "eval": eval_tokenized}

    torch.save(tokenized_data, "./wikitext-103-raw/tokenized_data.torch")
else:
    tokenized_data = torch.load(
        "./wikitext-103-raw/tokenized_data.torch", weights_only=True
    )


def get_batch(data, batchsize, length):
    starts = torch.randint(
        high=len(tokenized_data["train"]) - length - 1, size=(batchsize, 1)
    )
    indices = starts + torch.arange(length + 1)
    tokens = torch.take_along_dim(
        tokenized_data["train"], indices.reshape(-1), dim=0
    ).to(torch.int64)
    tokens = tokens.reshape([batchsize, -1])
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    return (inputs, targets)


"""
The gpt model is from 
    https://github.com/karpathy/nanoGPT/blob/master/model.py
"""


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(
                    -1, logits.size(-1)
                ),  # I changed to reshape; view doesn't work in my config
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


"""
AdamW vs PSGD-affine-whitening on a tiny gpt model.
We align their settings, and the only difference is preconditioner.   
"""

batchsize = 128
block_size = 128
num_iterations = 100_000
eval_every = num_iterations // 100
tinyConfig = GPTConfig(block_size=block_size, n_layer=6, n_head=12, n_embd=384)
tinyGpt = GPT(tinyConfig)


def test(data, model):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_trials = math.ceil(len(data) / (batchsize * block_size))
        for _ in range(num_trials):
            inputs, targets = get_batch(data, batchsize, block_size)
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = gpt(inputs, targets)
            total_loss += loss
    model.train()
    return total_loss.item() / num_trials


plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.yaxis.tick_right()
ax2.yaxis.tick_right()

"""
AdamW 
"""
decoupled_wd = 1e-2  # tiny model; no need of large wd
gpt = copy.deepcopy(tinyGpt).to(device)
opt = torch.optim.AdamW(gpt.parameters(), lr=1e-3, weight_decay=decoupled_wd)

TrainLoss = []
EvalLoss = []
t0 = time.time()
for num_iter in range(num_iterations):
    """train"""
    inputs, targets = get_batch(tokenized_data["train"], batchsize, block_size)
    inputs, targets = inputs.to(device), targets.to(device)

    def closure():
        _, loss = gpt(inputs, targets)
        return loss

    opt.zero_grad()
    loss = closure()
    loss.backward()
    opt.step()
    TrainLoss.append(loss.item())

    """test"""
    if (num_iter + 1) % eval_every == 0:
        EvalLoss.append(test(tokenized_data["eval"], gpt))
        print(f"AdamW, iter {num_iter + 1}, eval loss {EvalLoss[-1]}")

total_time = time.time() - t0

SmoothedTrainLoss = np.convolve(TrainLoss, np.ones(100)/100)[99:-99]
ax1.plot(
    torch.arange(1, len(SmoothedTrainLoss) + 1).to(torch.float32).cpu() * total_time / len(SmoothedTrainLoss),
    SmoothedTrainLoss,
)
ax2.plot(
    torch.arange(1, len(EvalLoss) + 1).to(torch.float32).cpu() * total_time / len(EvalLoss),
    EvalLoss,
)


"""
PSGD 
"""
gpt = copy.deepcopy(tinyGpt).to(device)

decoupled_wd = 1e-2  # keep the same setting as AdamW
opt = psgd.KronWhiten(
    gpt.parameters(),
    preconditioner_max_skew=2,
    momentum=0.9,
    lr_params=1e-3/4, # reduce adam lr by sqrt((1 + 0.9)/(1 - 0.9)) times with momentum 0.9
    grad_clip_max_amp=1.0,
    whiten_grad=False,
    preconditioner_update_probability=1.0, # anneal to 0.01
)

TrainLoss = []
EvalLoss = []
t0 = time.time()
for num_iter in range(num_iterations):
    """train"""
    inputs, targets = get_batch(tokenized_data["train"], batchsize, block_size)
    inputs, targets = inputs.to(device), targets.to(device)

    def closure():
        _, loss = gpt(inputs, targets)
        return loss

    with torch.no_grad():  # decoupled weight decay
        [
            p.subtract_(opt.lr_params * decoupled_wd * p)
            for p in opt._params_with_grad
        ]
    loss = opt.step(closure)
    TrainLoss.append(loss.item())

    """test"""
    if (num_iter + 1) % eval_every == 0:
        EvalLoss.append(test(tokenized_data["eval"], gpt))
        print(f"PSGD, iter {num_iter + 1}, eval loss {EvalLoss[-1]}")
       
        opt.preconditioner_update_probability = max(opt.preconditioner_update_probability * 0.1**0.1, 0.01)

total_time = time.time() - t0

SmoothedTrainLoss = np.convolve(TrainLoss, np.ones(100)/100)[99:-99]
ax1.plot(
    torch.arange(1, len(SmoothedTrainLoss) + 1).to(torch.float32).cpu() * total_time / len(SmoothedTrainLoss),
    SmoothedTrainLoss,
)
ax2.plot(
    torch.arange(1, len(EvalLoss) + 1).to(torch.float32).cpu() * total_time / len(EvalLoss),
    EvalLoss,
)


ax1.set_xlabel("Wall time (s)", fontsize=6)
ax1.set_ylabel("Train loss", fontsize=6)
ax1.tick_params(labelsize=6)
ax1.legend(
    [
        "Adam",
        r"PSGD, $dQ=Q^{0.5}\mathcal{E}Q^{1.5}$",
    ],
    fontsize=7,
)
ax1.set_ylim(min(SmoothedTrainLoss) - 0.1, 1 + min(SmoothedTrainLoss))
ax1.set_title("(a)", fontsize=7)

ax2.set_xlabel("Wall time (s)", fontsize=6)
ax2.set_ylabel("Validation loss", fontsize=6)
ax2.tick_params(labelsize=6)
ax2.legend(
    [
        "Adam",
        r"PSGD, $dQ=Q^{0.5}\mathcal{E}Q^{1.5}$",
    ],
    fontsize=7,
)
ax2.set_ylim(min(EvalLoss) - 0.1, 1 + min(EvalLoss))
ax2.set_title("(b)", fontsize=7)

plt.savefig("gpt2_adamw_vs_psgd.svg")
plt.savefig("gpt2_adamw_vs_psgd.eps")
plt.show()
