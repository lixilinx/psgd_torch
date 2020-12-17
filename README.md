## Pytorch implementation of PSGD  
### An overview
PSGD (preconditioned stochastic gradient descent) is a general purpose second-order optimization method. PSGD differentiates itself from most existing methods by its inherent abilities of handling nonconvexity and gradient noises. Please refer to the [original paper](https://arxiv.org/abs/1512.04202) for its designing ideas. Compared with the [old implementation](https://github.com/lixilinx/psgd_torch/releases/tag/1.0), this new Pytorch implementation greatly simplifies the usage of Kronecker product preconditioner. You may also refer to the updated [TensorFlow 2.x PSGD implementation](https://github.com/lixilinx/psgd_tf).
### Implemented preconditioners 
#### General purpose preconditioners
*Dense preconditioner*: this preconditioner is related to the classic Newton method. 

*Sparse LU decomposition preconditioner*: this one resembles the limited-memory BFGS method. 

*Diagonal preconditioner*: this reduces to the [equilibration preconditioner](https://arxiv.org/abs/1502.04390). The closed-form solution is available, and its implementation is trivial.  
#### Kronecker product preconditioners
For matrix parameters, we can have a left and a right preconditioner on its gradient. [This paper](https://openreview.net/forum?id=Bye5SiAqKX) discusses the design of such preconditioners in detail. Either the left or the right preconditioner can be a dense (resembles feature whitening), or a normalization (similar to batch normalization), or a scaling preconditioner. The code can switch to the right implementations by checking the dimensions of each preconditioner. 

For example, a left or right preconditioner with dimension [*N*, *N*] is dense; [2, *N*] is for normalization; and [1, *N*] for scaling. But, there is ambiguity when *N*=2 (a [2, 2] preconditioner can be either a dense or normalization type). Here, we always assume that a squared preconditioner is dense.    

### Implemented demos with classic problems
*hello_psgd.py*: a simple example on PSGD for Rosenbrock function minimization.

*mnist_with_lenet5.py*: demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. The test classification error rate of PSGD is significantly lower than other competitive methods like SGD, momentum, Adam, KFAC, etc. Please check the [archived code](https://github.com/lixilinx/psgd_torch/releases/tag/1.0) for details.  

*lstm_with_xor_problem.py*: demonstration of PSGD on gated recurrent neural network training with the delayed XOR problem proposed in the original [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this 'simple' problem with first order optimization method. PSGD is likely to solve it with either the LSTM or the simplest vanilla RNN (please check this [archived TF 1.x code](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) for details).

*demo_usage_of_all_preconditioners.py*: demonstrate the usage of all implemented preconditioners on the tensor decomposition math problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typical preconditioner initial guesses are (up to a positive scaling difference): identity matrix for a dense preconditioner; [[1,1,...,1],[0,0,...,0]] for normalization preconditioner; and [1,1,...,1] for scaling preconditioner.  

### Miscellaneous topics

*No support of higher order derivative for Hessian-vector product calculation*: some modules like Baidu's CTC implementation do not support higher order derivatives, and thus there is no way to calculate the Hessian-vector product exactly. However, you can use numerical method to approximate it as examples in this [archived TF 1.x code](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and the [original paper](https://arxiv.org/abs/1512.04202). Most likely, there is no big performance difference between the use of exact and approximated Hessian-vector products.  

*Which preconditioner to use?*: Dense preconditioner for small scaled problems (< 1e4 parameters); (dense, dense) Kronecker product preconditioners for most middle scaled problems, where the matrix size is about [1e3, 1e3]; (dense, normalization) or (normalization, dense) Kronecker product preconditioners from large scaled problems involving matrices with sizes up to [1e3, 1e6] or [1e6, 1e3], e.g., the language modeling example in [this paper](https://openreview.net/forum?id=Bye5SiAqKX); eventually, the (scaling, normalization) or (normalization, scaling) Kronecker product preconditioners is sufficiently sparse for matrices with sizes up to [1e6, 1e6] (really too large to be of any real use).

*NaN?* PSGD might have diverged. Try reducing the initial guess for preconditioner, or reducing the learning rate, or clipping the preconditioned gradient. When neither of these remedies works, check whether Hessian-vector products produce NaN first. Second-order derivatives under- or over-flow more easily than we thought, especially with single or half precision arithmetic. 

*Use of non-differentiable functions/modules*: theoretically, non-differentiable functions/modules lead to vanishing/undefined/ill-conditioned Hessian. PSGD does not use the Hessian directly. Instead, it just tries to align the distance metrics between the parameter and gradient spaces (resembles the Bregman divergence), and typically works well with such irregularities. For example, considering the LeNet5 model, both ReLU and max-pooling only have sub-gradients, still, PSGD works extremely well.  

*Reducing time complexity per step*: A simple trick is to update the preconditioner less frequently. Curvatures typically evolve slower than gradients. So, we have no need to update the preconditioner at every iteration.

*Reducing the spatial complexity*: PSGD needs more memory than SGD to calculate the Hessian-vector product. Still, you can use the numerical method to approximate the Hessian-vector products by only using the graph for gradient calculation. This reduces the memory consumption a lot. 

*Parameter 'step' for preconditioner update*: 0.01 works well for most stochastic optimization. Yet, it can be significantly larger for mathematical optimization as there is no gradient noise.

*Parameter '_tiny' for preconditioner update*: used solely to avoid division by zero. Just use the smallest positive normal number, e.g., about 1.2e-38 for torch.float32.
