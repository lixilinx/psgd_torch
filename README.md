## Pytorch implementation of PSGD 
*Major recent updates*: added gradient whitening preconditioners to all classes; wrapped affine preconditioners as a class (also support complex matrices); tighter lower bound for matrix spectral norm.
### An overview
PSGD (Preconditioned SGD) is a general purpose (mathematical or stochastic, convex or nonconvex) 2nd order optimizer. Unlike many 2nd order optimizers, PSGD does not rely on damping terms to stabilize it.    
#### A brief summary of the math. 
Notations: $H$ the Hessian; $h=Hv$ the Hessian-vector product; $g$ the gradient; $P=Q^TQ$ the preconditioner.

* Table I: Variations of preconditioner fitting loss
  
| Preconditioner fitting loss | Solution | Notes |
|--------------|------------------|-------|
|$h^TPh + v^TP^{-1}v$ | $Ph(Ph)^T = vv^T$ | Reduces to $Ph=v$ when  $H\succ 0$, i.e., secant equation for Quasi-Newton methods like BFGS. The default fitting loss in our implementations.  | 
| $E_{v\sim\mathcal{N}(0,I)}[h^TPh + v^TP^{-1}v] = {\rm tr}\left(PH^2 + P^{-1}\right)$ | $P=(H^2)^{-1/2}$ | Reduces to Newton method when  $H\succ 0$. Its diagonal solution $P={\rm diag}(1/\sqrt{E[h^2]})$ is rediscovered in many places, e.g., ESGD, AdaHessian, Sophia, etc.  | 
| $E_{v\sim\mathcal{N}(0,I)}[g^TPg + v^TP^{-1}v] = {\rm tr}\left(PE[gg^T] + P^{-1}\right)$ | $P=(E[gg^T])^{-1/2}$ | Reduces to gradient whitening methods, e.g., the AdaGrad family like Adam(W), RMAProp, Shampoo, etc. With per-sample gradient $g$, $P^{-2}$ reduces to the (empirical) Fisher or Gauss-Newton matrix.  | 

* Table II: Lie group examples for preconditioner fitting

|Preconditioner form|Group|Notes|
|-------------------|-----|-----|
| $Q\in \mathbb{R}^{N\times N}$, $\det(Q)>0$ | ${\rm GL}^{+}(N, \mathbb{R})$ or ${\rm Tri}^{+}(N, \mathbb{R})$ | Related to methods like Newton, BFGS, etc. Only practical with $N\le 10^4$. See our Newton preconditioner.  |
| $Q={\rm diag}(q_1, \ldots, q_N)$, $q_n>0$ $\forall$ $n$ | Diagonal subgroup | Related to methods with element-wise preconditioning (the most popular choice).    |
| $Q=\oplus_{\ell=1}^L  \left[ (Q_{2, \ell}^TQ_{2, \ell}) \otimes (Q_{1, \ell}^TQ_{1, \ell}) \right] $  | Each $Q_{i,\ell}$'s group can be different, typically affine group  | Related to KFAC, Shampoo, etc. Our implementation also supports complex matrices, and $Q_{i,\ell}$ can be diagonal to save resources if too large. See our Affine preconditioner. |
| $Q = UV^T + {\rm diag}(d_1, \ldots, d_N)$, $U$ and $V$ $\in \mathbb{R}^{N\times r}$, $d_n>0$, $\det(Q)>0$, and $0\le r\ll N$ | [Here](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for details | Related to LM-BFGS, nonlinear CG, etc. See our LRA (low-rank approximation) preconditioner. |      
  
#### Highlights of PSGD 
##### Normalized step sizes 
The two learning rates (lr) of PSGD are self-normalized to range [0, 1] as below:
* The lr for preconditioner fitting is normalized as ${\rm lr} ||\nabla_Q||<1$ such that $Q$ always lies on the same connected Lie group ([demo code](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py)).
* The lr for parameter learning already is normalized as in the Newton method since $P = (H^2)^{-1/2}$.
   
##### Quadratic convergence
See the [Rosenbrock function demo](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py) for a toy example.   
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/quadratic_convergence.svg" width=40% height=40%>

##### Efficiency
PSGD is cheaper per step than classic methods like BFGS as it does not rely on line search. See the [tensor rank decomposition benchmark](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py) for the following typical comparison results.      

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_vs_bfgs.svg" width=40% height=40%>

##### Numerical stability
All the PSGD implementations are stable with single and half (bfloat16, not float16 due to its small dynamic range) precisions, without fiddling with any damping factors. [This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) compares PSGD and the closed-form solutions in Table I to have the following sample results. PSGD is efficient by solving the secant equation directly, and also free of numerically problematic operations like matrix inverse, eigenvalue decompositions, etc.    

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=50% height=50%>

##### Generalization property
Abundant empirical results suggest that PSGD generalizes as well as or better than its kin SGD. [This LeNet5 demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py) is a good toy example illustrating this point in the view of information theory. Starting from the same initial guesses, PSGD tends to find minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. Indeed, minima with shorter DLs tend to give better test performance, as suggested by arrows pointing to the down-left-front corner of the cube.           
<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.svg" width=50% height=50%>

### Implemented preconditioners 
#### Three families of preconditioners
##### Matrix-free preconditioners 
We can construct sparse preconditioners from subgroups of the permutation group. Theoretically, they just reduce to the direct sum of smaller groups. Specifically, subgroup {e} induces the diagonal or Jacobi preconditioner, possibly the most popular choice. This simplest form of PSGD actually works on many problems ([benchmarked here](https://github.com/lixilinx/psgd_tf/releases/tag/1.3)). Subgroup {e, flipping} induces the X-shape matrices. Subgroup {e, half_len_circular_shifting} induces the butterfly/Kaleidoscope matrices. Many possibilities. I only have implemented the preconditioner on the group of X-matrix. 

##### Low-rank approximation (LRA) preconditioner
This group has form $Q = UV^T+{\rm diag}(d)$, thus simply called the UVd preconditioner too. Its hidden Lie group geometric structure is revealed [here](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view?usp=sharing). Standard LRA forms, e.g., $P = {\rm diag}(d) + UU^T$, can only fit one end of the spectra of Hessian. This form can fit both ends. Hence, a low-order approximation may work well for problems with millions of parameters. The LRA preconditioner is of great practical value since it is easy to use and performs great. Hessians and their inverses from most practical problems tend to have an extremely low rank structure.

##### Affine group preconditioners
Most neural networks consist of affine transformations and simple activation functions. Each affine transformation matrix can have a left and a right preconditioner on two affine groups correspondingly. This is a rich family, including the preconditioner forms in KFAC, Shampoo, and batch, or layer, or group normalizations, and many more as special cases, although conceptually different. PSGD is a more widely applicable and flexible framework, and also free of numerically problematic operations like matrix inverse, eigenvalue decomposition, etc.

#### Implementation details 
These preconditioners are wrapped into classes *XMat*, *LRA* (or *UVd*), *Newton* and *Affine* for easy use. The affine family are not black box preconditioners, and the users need to either group the gradients into matrices or reform the models such that the parameters are a list of matrices, e.g., [this demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) on wrapping torch.nn.functional.conv2d as an affine Conv2d class, and [this one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class. Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivative is not available.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding $L2$ regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementations.    

### Demos on a few classic problems
[Rosenbrock function](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): a simple example of PSGD on the Rosenbrock function minimization.

[LeNet5](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation. 
 
[Delayed XOR](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this "naive" problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[Tensor rank decomposition](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all preconditioners on the tensor rank decomposition problem. PSGD performs no worse than classic methods like BFGS here.     

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large-scale logistic regression problem. PSGD performs no worse than LM-BFGS, "the algorithm of choice" for such problems. 

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Other implementations: [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf). I no longer maintain them as I have not used Tensorflow for 1+ years.
