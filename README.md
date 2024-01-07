## Pytorch implementation of PSGD 
[Dec 2023 updates](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_updates.pdf): added gradient whitening preconditioners to all classes; wrapped affine preconditioners as a class (also support complex matrices); tighter lower bound for triangular matrix norm; LRA preconditioner with rank 0 reduces to diagonal preconditioners. 
### An overview
PSGD (preconditioned SGD) is a general purpose 2nd order optimizer. PSGD differentiates itself from most methods by its inherent ability to handle nonconvexity and gradient noises. Now I have wrapped most preconditioners (Kronecker product or affine, low-rank approximation (LRA), matrix-free and Newton) as classes for easy use. There are [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf) implementations as well, but not actively maintained. The two learning rates (lr) of PSGD are always normalized into range [0, 1] as below:

1) lr for preconditioner fitting. The preconditioner (its factor Q) is updated on a one-parameter, i.e., lr, subgroup. Thus, the lr should be normalized as || lr*G ||<1 such that Q always is on the same connected Lie group ([demo code](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py)).
2) lr for parameter learning. This lr already is normalized as in the Newton method since P = inv(abs(H)) (indeed, PSGD reduces to the Newton method for convex optimization).    

*Toy example on quadratic convergence*: Not a surprise as PSGD recovers the Newton method. See the [Rosenbrock function demo](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py).   
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/quadratic_convergence.svg" width=40% height=40%>

*Toy example on generalization property*: Abundant of empirical results have confirmed that PSGD generalizes as well as or better than its kin SGD. [This LeNet5 demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py) is a good toy example illustrating this in the view of information theory. Starting from the same random initial guesses, PSGD converges to minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. Indeed, minima with shorter DLs tend to give better test performance, as suggested by arrows pointing to the down-left-front corner of the cube.           
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.svg" width=50% height=50%>

*Numerical stability*: All the PSGD implementations are extremely numerically stable with single and half (bfloat16, not float16 due to its small dynamic range) precisions. The equivariance property of Lie group implies that PSGD is actually fitting P*H, which is eventually close to I, regardless of cond(H). This remarkable property saves the need of any damping required by other 2nd order optimizers. [This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) compares PSGD and closed-form solutions to have the following sample results. It is not a surprise that PSGD could give higher quality preconditioner estimation as it estimates P directly, not to mention the numerical errors in closed-form solutions, say, due to eigenvalue decompositions (EVD).    

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=50% height=50%>

### Implemented preconditioners on a few families of Lie groups 
I categorize them into three families. 

*Type I): Matrix-free preconditioners*. We can construct sparse preconditioners from subgroups of the permutation group. Theoretically, they just reduce to the direct sum of smaller groups. But, practically, they can perform well by shortcutting gradients far away in positions.    

Subgroup {e} induces the diagonal or Jacobi preconditioner. PSGD reduces to the [equilibrated SGD (ESGD)](https://arxiv.org/abs/1502.04390) and [AdaHessian](https://arxiv.org/abs/2006.00719). This simplest form of PSGD actually works on many problems ([benchmarked here](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)). 

Subgroup {e, flipping} induces the X-shape matrices. Subgroup {e, half_len_circular_shifting} induces the butterfly/Kaleidoscope matrices. Many possibilities. I only have implemented the preconditioner on the group of X-matrix. 

*Type II): Low-rank approximation (LRA) preconditioner*. This group has form Q = U*V'+diag(d), thus simply called the UVd preconditioner too. Its hidden Lie group geometric structure is revealed [here](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view?usp=sharing).   

Standard LRA form, e.g., P = diag(d) + U*U', can only fit one end of the spectra of Hessian. This form can fit both ends. Hence, a low-order approximation may work well for problems with millions of parameters. The LRA preconditioner is of great practical value since it is easy to use and performs great. Hessians and their inverses from most practical problems tend to have an extremely low rank structure.

*Type III): Affine group preconditioners*. Most neural networks consist of affine transformations and simple activation functions. Each affine transformation matrix can have a left and a right preconditioner on two affine groups correspondingly. This is a rich family, including the preconditioner forms in KFAC, Shampoo, and batch, or layer, or group normalizations, and many more as special cases, although conceptually different. PSGD is a more widely applicable and flexible framework, and free of numerical issues from matrix inverse or EVD.

Most of these preconditioner are wrapped as classes for easy use (XMat, LRA or UVd, Newton and Affine). The affine family are not black box preconditioners, and the users need to either group the gradients into matrices or reform the models such that the parameters are a list of matrices, e.g., [this demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) on wrapping torch.nn.functional.conv2d as an affine Conv2d class, and [this one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class. Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivative is not available.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding L2 regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementation.    

### Implemented demos on a few classic problems
[Rosenbrock function](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): a simple example of PSGD on *Rosenbrock function* minimization.

[LeNet5](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): demonstration of PSGD on convolutional neural network training with the classic *LeNet5* for MNIST digits recognition. The test classification error rate of PSGD (<0.7%) is significantly lower than other competitive methods like SGD, momentum, Adam, KFAC, etc. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation. 
 
[Delayed XOR](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the *delayed XOR problem* proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this 'naive' problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[CPD](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all Kronecker preconditioners on the tensor rank decomposition, i.e., *canonical polyadic decomposition (CPD)*, problem. Note that all kinds of Kronecker product preconditioners share the same way of usage. You just need to pay attention to its initializations. Typical (either left or right) preconditioner initial guesses are (up to a scaling difference): identity matrix for a dense or whitening preconditioner; [[1,1,...,1],[0,0,...,0]] for a normalization preconditioner; and [1,1,...,1] for a scaling preconditioner. Now I have wrapped the affine family as a class to hide all these details.    

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large scale logistic regression problem. PSGD also performs well. 

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
