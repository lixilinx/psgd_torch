## Pytorch implementation of PSGD 

### An overview
PSGD (preconditioned stochastic gradient descent) is a general purpose 2nd optimization method. PSGD differentiates itself from most methods by its inherent ability of handling nonconvexity and gradient noises. [This paper](https://arxiv.org/abs/1512.04202) documents its design ideas. Compared with the [old implementation](https://github.com/lixilinx/psgd_torch/releases/tag/1.0), the new one simplifies the usage of Kronecker product preconditioner, and introduce new black box, e.g., low-rank approximation (LRA), preconditioners. There are [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf) implementations as well, but not actively maintained.

*Quadratic convergence*: Not a surprise as a 2nd method. See the [Rosenbrock function demo](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py).   
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/quadratic_convergence.svg" width=40% height=40%>

*Generalization property*: The way PSGD handling gradient noises let it generalize as well as its kin SGD. [This LeNet5 demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py) is a good toy example illustrating this. Starting from the same initial points, PSGD converges to minima with smaller train data cross entropy and smaller/flatter Hessians than Adam, i.e., shorter description lengths (DL) for data and model. Indeed, solutions with shorter DLs give better test performance (arrows point to the down-left corner of the plot).           
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.svg" width=40% height=40%>

### Implemented preconditioners on Lie groups 
I categorize them into three families. 

*Type I): Matrix free preconditioners*. We can construct sparse preconditioners from subgroups of the permutation group. Theoretically, they just reduce to the direct sum of smaller groups. But, practically, they can perform well by shortcutting gradients far away in positions.   

Subgroup {e} induces the diagonal/Jacobi preconditioner. PSGD reduces to the [equilibrated SGD (ESGD)](https://arxiv.org/abs/1502.04390) and AdaHessian exactly. It works, but not good enough without the help of momentum ([benchmarked here](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)). 

Subgroup {e, flipping} induces the X-shape matrices. Subgroup {e, half_len_circular_shifting} induces the butterfly/Kaleidoscope matrices. Many possibilities. Only the subgroup {e, flipping} is implemented for now.  

*Type II): Low-rank approximation (LRA) preconditioner*. This group has form Q = U*V'+diag(d), thus simply called the UVd preconditioner. Its hidden Lie group geometric structure is disclosed [here](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view?usp=sharing).   

Standard LRA form, e.g., P = diag(d) + U*U', can only fit one end of the spectra of Hessian. This form can fit both ends. Hence, a low order approximation could work well for problems with millions of parameters. 

*Type III): Affine group preconditioners*. Most neural networks consist of affine transformations and simple activation functions. Each affine transformation matrix can have a left and a right preconditioner on two affine groups seperately. This is a rich family, including the preconditioner forms in KFAC, and batch and layer normalizations as special cases, although conceptually different. PSGD is a far more widely applicable and flexible framework. [Further details](https://openreview.net/forum?id=Bye5SiAqKX).

The first two families and the classic Newton–Raphson like preconditioner are wrapped as classes for easy use (XMat, UVd and Newton). But the affine family are not black box preconditioners, and thus I only give their functional form implementations. Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation via finite-difference when 2nd derivative is not available;   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD; 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding L2 regularization to the loss in the closure. Similarly, decoupled weight decay is not included inside PSGD.    

### Implemented demos with classic problems
*hello_psgd.py*: a simple example on PSGD for Rosenbrock function minimization.

*mnist_with_lenet5.py*: demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. The test classification error rate of PSGD is significantly lower than other competitive methods like SGD, momentum, Adam, KFAC, etc. [Archived benchmarks](https://github.com/lixilinx/psgd_torch/releases/tag/1.0).  

*lstm_with_xor_problem.py*: demonstration of PSGD on gated recurrent neural network with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this 'simple' problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN ([archived TF 1.x code for benchmarks](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)).

*demo_usage_of_all_preconditioners.py*: demonstrate the usage of all Kronecker preconditioners on the tensor rank decompositionthe, i.e., canonical polyadic decomposition (CPD), problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typical (either left or right) preconditioner initial guesses are (up to a scaling difference): identity matrix for a dense/whitening preconditioner; [[1,1,...,1],[0,0,...,0]] for a normalization preconditioner; and [1,1,...,1] for a scaling preconditioner.  

*misc*: logistic regression, cifar10, and more. 

### Ref
Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015.
