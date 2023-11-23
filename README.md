## Pytorch implementation of PSGD 

### An overview
PSGD (preconditioned stochastic gradient descent) is a general purpose 2nd order optimizer. PSGD differentiates itself from most methods by its inherent ability to handle nonconvexity and gradient noises. Building on top of the [old implementation](https://github.com/lixilinx/psgd_torch/releases/tag/1.0), the new one simplifies the usage of Kronecker product preconditioner (on affine group), and introduces new black box, e.g., the low-rank approximation (LRA), preconditioners. The LRA one is of great practical value since it is easy to use and performs great. There are [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf) implementations as well, but not actively maintained. The two learning rates (lr) of PSGD are always normalized into range [0, 1] as below:

1) lr for preconditioner fitting. Essentially, at each step, the preconditioner (its Cholesky factor Q) is updated on a one-parameter, i.e., lr, subgroup. Thus, the lr should be normalized as ||lr_normalized*G||<1 such that Q always is on the same connected Lie group.
2) lr for parameter learning. The preconditioner matches metrics in the gradient and parameter spaces as ||delta(P*g)|| = ||delta(params)||. Thus lr already is normalized as in the Newton method (indeed, PSGD reduces to the Newton method for convex optimization).    

*Toy example 1 on quadratic convergence*: Not a surprise as PSGD recovers the Newton method. See the [Rosenbrock function demo](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py).   
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/quadratic_convergence.svg" width=40% height=40%>

*Toy example 2 on generalization property*: Abundant of empirical results have confirmed that PSGD generalizes as well as or better than its kin SGD. [This LeNet5 demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py) is a good toy example illustrating this in the view of information theory. Starting from the same random initial network guesses, PSGD converges to minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. Indeed, minima with shorter DLs tend to give better test performance, as suggested by arrows pointing to the down-left-front corner of the cube.           
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.svg" width=50% height=50%>

### Implemented preconditioners on a few families of Lie groups 
I categorize them into three families. 

*Type I): Matrix-free preconditioners*. We can construct sparse preconditioners from subgroups of the permutation group. Theoretically, they just reduce to the direct sum of smaller groups. But, practically, they can perform well by shortcutting gradients far away in positions.    

Subgroup {e} induces the diagonal or Jacobi preconditioner. PSGD reduces to the [equilibrated SGD (ESGD)](https://arxiv.org/abs/1502.04390) and [AdaHessian](https://arxiv.org/abs/2006.00719) exactly. This simplest form of PSGD actually works well on many problems ([benchmarked here](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)). 

Subgroup {e, flipping} induces the X-shape matrices. Subgroup {e, half_len_circular_shifting} induces the butterfly/Kaleidoscope matrices. Many possibilities. I only have implemented the preconditioner on the group of X-matrix. 

*Type II): Low-rank approximation (LRA) preconditioner*. This group has form Q = U*V'+diag(d), thus simply called the UVd preconditioner. Its hidden Lie group geometric structure is revealed [here](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view?usp=sharing).   

Standard LRA form, e.g., P = diag(d) + U*U', can only fit one end of the spectra of Hessian. This form can fit both ends. Hence, a low-order approximation could work well for problems with millions of parameters. The LRA preconditioner is of great practical value since it is easy to use and performs great. Hessians and their inverses from most practical problems tend to have an extremely low rank structure.

*Type III): Affine group preconditioners*. Most neural networks consist of affine transformations and simple activation functions. Each affine transformation matrix can have a left and a right preconditioner on two affine groups correspondingly. This is a rich family, including the preconditioner forms in KFAC, and batch, or layer, or group normalizations, and many more as special cases, although conceptually different. PSGD is a far more widely applicable and flexible framework. [Further details](https://openreview.net/forum?id=Bye5SiAqKX) here.

The first two families and the classic Newton–Raphson preconditioner are wrapped as classes for easy use (XMat, UVd and Newton). The affine family are not black box preconditioners, and thus I only give their functional form implementations. Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite differences when the 2nd derivative is not available.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding L2 regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementation.    

### Implemented demos on a few classic problems
*hello_psgd.py*: a simple example of PSGD on *Rosenbrock function* minimization.

*mnist_with_lenet5.py*: demonstration of PSGD on convolutional neural network training with the classic *LeNet5* for MNIST digits recognition. The test classification error rate of PSGD is significantly lower than other competitive methods like SGD, momentum, Adam, KFAC, etc. See [archived benchmarks](https://github.com/lixilinx/psgd_torch/releases/tag/1.0).  

*lstm_with_xor_problem.py*: demonstration of PSGD on gated recurrent neural network (RNN) learning with the *delayed XOR problem* proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this 'naive' problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (see [archived TF 1.x code for benchmarks](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)).

*demo_usage_of_all_preconditioners.py*: demonstrate the usage of all Kronecker preconditioners on the tensor rank decomposition, i.e., *canonical polyadic decomposition (CPD)*, problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typical (either left or right) preconditioner initial guesses are (up to a scaling difference): identity matrix for a dense or whitening preconditioner; [[1,1,...,1],[0,0,...,0]] for a normalization preconditioner; and [1,1,...,1] for a scaling preconditioner.  

*misc*: logistic regression, cifar10, and more. 

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (The general theory of PSGD.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners on the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the low-rank approximation (LRA) preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for readers without Lie group background.)
