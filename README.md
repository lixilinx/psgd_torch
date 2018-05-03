### Pytorch implementation of PSGD 
Verified on Pytorch 0.4.0. Please try 'hello_psgd.py' first to see whether it works for you. We have included one RNN learning benchmark problem, and five demos showing the usage of five different preconditioners. For more information on PSGD, please check https://github.com/lixilinx/psgd_tf

Note: *torch.trtrs is not implemented for CUDA tensors yet. So currently, preconditioners using this function may not work on GPU.*
