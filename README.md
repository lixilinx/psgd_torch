## Pytorch implementation of PSGD 
*Major recent updates*: added gradient whitening preconditioners to all classes; wrapped affine preconditioners as a class (also support complex matrices); preconditioner fitting step size normalization with the 2nd derivative info.
### An overview
PSGD (Preconditioned SGD) is a general purpose (mathematical or stochastic, convex or nonconvex) 2nd order optimizer. Unlike many 2nd order optimizers, PSGD does not rely on damping terms to stabilize it.    

Notations: $H$ the Hessian; ${v\sim\mathcal{N}(0,I)}$; $h=Hv$ the Hessian-vector product; $g$ the gradient; $P=Q^TQ$ the preconditioner.

#### Table I: Variations of preconditioner fitting loss
  
| Fitting loss of $P$ | Solution | Notes |
|--------------|------------------|-------|
|$h^TPh + v^TP^{-1}v$ | $Phh^TP = vv^T$ | Reduces to secant equation $Ph=v$ when  $v^Th>0$ (Quasi-Newton methods, e.g., BFGS); set preconditioner_type="Newton" for this implementation.   | 
| $E_v[h^TPh + v^TP^{-1}v]$ | $P^{-2}=H^2$ | Reduces to Newton's method when  $H\succ 0$; set preconditioner_type="Newton" for this implementation.  | 
| $E_v[g^TPg + v^TP^{-1}v]$ | $P^{-2}=E[gg^T]$ | $P^{-2}$ reduces to Fisher information matrix with sample-wise gradient $g$ (Gauss-Newton and natural gradient family, e.g., KFAC); *no class implementation for now*.  | 
| $\sum_t E_{v_t}[g_t^TPg_t + v_t^TP^{-1}v_t]$ | $P^{-2}=\sum_t g_t g_t^T$ | Reduces to AdaGrad family, e.g., Adam(W), RMSProp, Shampoo, $\ldots$; set preconditioner_type="whitening" for this implementation.  | 

#### Table II: Implemented preconditioners on Lie groups 

|Lie Group|Update of $Q$ ($0<\mu\le 1$) |Notes|
|-------------------|-----|-----|
| ${\rm GL}(n, \mathbb{R})$  | $Q\leftarrow \left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \|\|Qh\|\|^2 + \|\| Q^{-T}v\|\|^2  } \right)   Q$ | See class *Newton*; set keep_invQ=True to calculate $Q^{-1}$ recursively via Woodbury formula.   |
| Triangular matrices | $Q\leftarrow \left[ I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \|\|Qh\|\|^2 + \|\| Q^{-T}v\|\|^2  } \right]_R   Q$ | See class *Newton*; set keep_invQ=False to make $Q$ triangular and calculate $Q^{-T}v$ with backward substitution; $[I+A]_R$ only keeps $R$ of the QR decomposition of $I+A$ and $[I+A]_R\approx {\rm triu}(A) + {\rm triu}(A,1)$ for $\|\|A\|\| \ll 1$.     |
| Diagonal matrices | $Q\leftarrow \left( I - \mu \frac{(Qh)^2 - (v/Q)^2}{  \max\left((Qh)^2 + (v/Q)^2\right)} \right)   Q$ | $Q$ is a vector; see classes *LRA* with rank_of_approximation=0 and *XMat* for implementations.  |
| $Q=\oplus_i(\otimes_j Q_{i,j})$, e.g., $Q=Q_2\otimes Q_1$ | $A=Q_1 hQ_2^H$, $B=Q_2^{-H}v^H Q_1^{-1}$, $Q_1\leftarrow \left[ I - \mu \frac{AA^H-B^HB}{\|\|AA^H+B^HB\|\|_F^2} \right]_R   Q_1$, $Q_2\leftarrow \left[ I - \mu \frac{A^HA-BB^H}{\|\|A^HA+BB^H\|\|_F^2} \right]_R   Q_2$ | Both $v$ and $h$ are matrices; see class *Affine* for implementations (also support complex matrices and diagonal matrices by setting preconditioner_max_size=0).  | 
| $Q=(I+UV^T){\rm diag}(d)$ | $a=Qh$, $b=Q^{-T}v$, $d\leftarrow \left(1 - \mu\frac{(Q^Ta)h-v(Q^{-1}b)}{\sqrt{\left((Q^Ta)^2 + v^2\right)\left(h^2 + (Q^{-1}b)^2\right)}}\right)d$, $U\leftarrow U - \mu\frac{(aa^T-bb^T)V(I+V^TU)}{\|\|a\|\| \\, \|\|VV^Ta \|\| + \|\|b\|\|\\, \|\|VV^Tb\|\|}$, $V\leftarrow V - \mu\frac{ (I+VU^T)(aa^T-bb^T)U }{\|\|a\|\| \\, \|\|UU^Ta\|\| + \|\|b\|\| \\, \|\|UU^Tb\|\|}$ | See class *LRA* for implementations; typically $0\le r\ll n$ with $U, V \in \mathbb{R}^{n\times r}$. |    

#### Preconditioner fitting accuracy   

[This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) generates the following plot showing the typical behaviors of different preconditioner fitting methods. 

* With a static and noise-free Hessian-vector product model, both BFGS and PSGD converge linearly to the optimal preconditioner while closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ only converges sublinearly with rate $\mathcal{O}(1/t)$.
* With a static additive noisy Hessian-vector model $h=Hv+\epsilon$, BFGS diverges easily, and PSGD may need a proper scheduler for $\mu$ (what shown here is with a constant $\mu$) to match the convergence speed and accuracy of closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$.
* With a time-varying Hessian $H_{t+1}=H_t + uu^T$ and $u\sim\mathcal{U}(0,1)$, PSGD locks onto good preconditioner estimations faster than BFGS.     

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=90% height=90%>

#### Implementation details 
The methods in Table 1 and preconditioners in Table 2 are wrapped into classes *XMat*, *LRA* (or *UVd*), *Newton* and *Affine* for easy use. The affine family are not black box preconditioners, and the users need to either group the gradients into matrices or reform the models such that the parameters are a list of matrices, e.g., [this demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) on wrapping torch.nn.functional.conv2d as an affine Conv2d class, and [this one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class. Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivative is not available.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding $L2$ regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementations.

### Demos 

[Rosenbrock function minimization](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): see how simple to apply PSGD to convex and stochastic optimizations. Typically, we just set preconditioner_init_scale=None, and may need to tune lr_params and lr_preconditioner a little. 

[LeNet5](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation. 
 
[Delayed XOR](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this "naive" problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[Tensor rank decomposition benchmark](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all preconditioners on the tensor rank decomposition problem. PSGD performs better than BFGS here.     

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large-scale logistic regression problem. PSGD performs better than LM-BFGS, "the algorithm of choice" for logistic regression.  

[Preconditioner fitting on Lie groups](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py): see how multiplicative updates work on Lie groups. 

[Preconditioner estimation efficiency and numerical stability](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py): a playground to compare PSGD with BFGS and closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$. Eigenvalue decompositions required by the closed-form solution can be unstable with single precisions, while PSGD is free of numerically problematic operations like matrix inverse, eigenvalue decompositions, etc.

[How PSGD generalizes no worse than SGD](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py): This one serves as a good toy example illustrating it in the view of information theory. Starting from the same initial guesses, PSGD tends to find minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. 

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Stochastic Hessian fitting on Lie group, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858). (Some ongoing theoretical works leading to the recent updates. The Hessian fitting is shown to be strongly convex on group ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Other implementations: [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf). I no longer maintain them as I have not used Tensorflow for some time.
