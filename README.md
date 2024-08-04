## Pytorch implementation of PSGD 
*Major recent updates*: added gradient whitening preconditioners to all classes; wrapped affine preconditioners as a class (also support complex and non-2D tensors); preconditioner fitting step size normalization with the 2nd derivative info.
### An overview
PSGD (Preconditioned SGD) is a general purpose (mathematical and stochastic, convex and nonconvex) 2nd order optimizer. It reformulates a wide range of preconditioner estimation and Hessian fitting problems as a family of strongly convex Lie group optimization problems. 

Notations: $E_z[\ell(\theta, z)]$ or $\ell(\theta)$ the loss;  $g$ the (stochastic) gradient wrt $\theta$; $H$ the Hessian;  $h=Hv$ the Hessian-vector product with ${v\sim\mathcal{N}(0,I)}$; $P=Q^TQ$ the preconditioner applying on $g$; $[I+A]_R\approx I + {\rm triu}(A) + {\rm triu}(A,1)$ for $\|\|A\|\| < 1$, where $[\cdot]_R$ keeps $R$ of a QR decomposition.

#### Table I: Variations of preconditioner fitting criterion
  
| Criterion | Solution | Notes |
|--------------|------------------|-------|
|$h^TPh + v^TP^{-1}v$ | $Phh^TP = vv^T$ | Reduces to secant equation $Ph=v$ when  $v^Th>0$ (Quasi-Newton methods, e.g., BFGS); set preconditioner_type="Newton" for this implementation.   | 
| $E_v[h^TPh + v^TP^{-1}v]$ | $P^{-2}=H^2$ | Reduces to Newton's method when  $H\succ 0$; set preconditioner_type="Newton" for this implementation.  | 
| $E_{v,z}[g^TPg + v^TP^{-1}v]$ | $P^{-2}=E_z[gg^T]$ | $P^{-2}$ reduces to Fisher information matrix $F$ with sample-wise gradient $g$ (Gauss-Newton and natural gradient family, e.g., KFAC); *no class implementation* (simply applying $P$ twice to get $F^{-1}$). | 
| $\sum_t E_{v_t}[g_t^TPg_t + v_t^TP^{-1}v_t]$ | $P^{-2}=\sum_t g_t g_t^T$ | Reduces to AdaGrad family, e.g., Adam(W), RMSProp, Shampoo, $\ldots$; set preconditioner_type="whitening" for this implementation.  |   

Note that $v$ can be a nuisance or an auxiliary variable in the last two criteria since it is independent of $g$ and $E_{v\sim\mathcal{N}(0,I)}[v^TP^{-1}v]={\rm tr}(P^{-1})$.   

#### Table II: Implemented Lie group preconditioners with storage and computation numbers for $\theta={\rm vec}(\Theta)$ and $\Theta\in\mathbb{R}^{m\times m}$ 

|Lie Group|Update of $Q$ ($0<\mu\le 1$)  |Storages | Computations |Notes|
|---|---|---|---|---|
| ${\rm GL}(n, \mathbb{R})$  | $Q\leftarrow \left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \|\|Qh\|\|^2 + \|\| Q^{-T}v\|\|^2  } \right)   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^4)$ | See class *Newton*; set keep_invQ=True to calculate $Q^{-1}$ recursively via Woodbury formula.   |
| Triangular matrices | $Q\leftarrow \left[ I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \|\|Qh\|\|^2 + \|\| Q^{-T}v\|\|^2  } \right]_R   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^6)$ | See class *Newton*; set keep_invQ=False to make $Q$ triangular and calculate $Q^{-T}v$ with backward substitution.     |
| Diagonal matrices, $Q={\rm diag}(q)$ | $q\leftarrow \left( I - \mu \frac{(qh)^2 - (v/q)^2}{  \max\left((qh)^2 + (v/q)^2\right)} \right)   q$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^2)$ | See class either *LRA* with rank_of_approximation=0 or *XMat* for implementations.  |
| $Q=\oplus_i(\otimes_j Q_{i,j})$, e.g., $Q=Q_2\otimes Q_1$ | $A=Q_1 {\rm uvec}(h) Q_2^H$, $B=Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1}$, $Q_1\leftarrow \left[ I - \mu \frac{AA^H-B^HB}{\|\|A\|\|_F^2 +\|\|B\|\|_F^2} \right]_R   Q_1$, $Q_2\leftarrow \left[ I - \mu \frac{A^HA-BB^H}{\|\|A\|\|_F^2 +\|\|B\|\|_F^2} \right]_R   Q_2$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ | See class *Affine* for implementations (also support complex and non-2D tensors, and diagonal $Q$).  | 
| $Q=(I+UV^T){\rm diag}(d)$ | $a=Qh$, $b=Q^{-T}v$, $d\leftarrow \left(1 - \mu\frac{(Q^Ta)h-v(Q^{-1}b)}{\max\sqrt{\left((Q^Ta)^2 + v^2\right)\left(h^2 + (Q^{-1}b)^2\right)}}\right)d$, $U\leftarrow U - \mu\frac{(aa^T-bb^T)V(I+V^TU)}{\|\|a\|\| \\, \|\|VV^Ta \|\| + \|\|b\|\|\\, \|\|VV^Tb\|\|}$, $V\leftarrow V - \mu\frac{ (I+VU^T)(aa^T-bb^T)U }{\|\|a\|\| \\, \|\|UU^Ta\|\| + \|\|b\|\| \\, \|\|UU^Tb\|\|}$ | $\mathcal{O}(rm^2)$ | $\mathcal{O}(rm^2)$ | See class *LRA* for implementations; typically $0\le r\ll n$ with $U, V \in \mathbb{R}^{n\times r}$. Update either $U$ or $V$, not both, per step.  |  
| Scaling or normalization | Similar to *Affine*, but $Q_1$ or $Q_2$ only for scaling or normalization. | $\mathcal{O}(m)$ | $\mathcal{O}(m^2)$ | With *Affine*, small preconditioner_max_size or preconditioner_max_skew makes large $Q$ diagonal. | 

For AdaGrad like gradient whitening preconditioner, we simply replace pair $(v, h)$ with $(v, g)$, where $v$ is a dummy variable that can be optionally integrated out. Taking the gradient whitening affine preconditioner as an example, we can integrate out $v$ as 

$E_{v\sim\mathcal{N}(0,I)}[BB^H] = E_v \left(Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1} Q_1^{-H} [{\rm uvec}(v)] Q_2^{-1}\right) ={\rm tr}\left(Q_2^{-1}Q_2^{-H}\right) \\, Q_1^{-H}Q_1^{-1}$,  $\\;$ $E_v[B^HB] = {\rm tr}\left(Q_1^{-1}Q_1^{-H}\right) \\, Q_2^{-H}Q_2^{-1}$  

Yet, keeping $v$ as an auxiliary variable simplifies the calculations most of the time.  

#### Preconditioner fitting accuracy   

[This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) generates the following plot showing the typical behaviors of different preconditioner fitting methods. 

* With a static and noise-free Hessian-vector product model, both BFGS and PSGD converge linearly to the optimal preconditioner while closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ only converges sublinearly with rate $\mathcal{O}(1/t)$.
* With a static additive noisy Hessian-vector model $h=Hv+\epsilon$, BFGS diverges easily. With a constant step size $\mu$, the steady-state fitting errors of PSGD are proportional  to $\mu$. 
* With a time-varying Hessian $H_{t+1}=H_t + uu^T$ and $u\sim\mathcal{U}(0,1)$, PSGD locks onto good preconditioner estimations quicker than BFGS, also no divergence before convergence. The closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ is not good at tracking due to its sublinear convergence.       

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=90% height=90%>

#### Implementation details 
Optimizers with the criteria in Table I and preconditioner forms in Table II are wrapped into classes *XMat*, *LRA* (or *UVd*), *Newton* and *Affine* for easy use. The affine family are not strictly black box preconditioners as it expects the optimization of a list of affine transform matrices. Thus, the Affine preconditioner reshapes any non-2D shaped parameters into matrices. 

Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivatives are unavailable.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding $L2$ regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementations. We recommend to randomize the regularization term, e.g., replacing the $L2$ one for a parameter $p$,  $0.5 \lambda \cdot {\rm sum}(p^2)$, with ${\rm rand}() \cdot \lambda\cdot {\rm sum}(p^2)$. 

A few more details. The Hessian-vector products are calculated as a vector-jacobian-product (vjp), i.e., ${\rm autograd.grad}(g, \theta, v)$ in torch, maybe not the most efficient way for a specific problem (see [this](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)). For gradient whitening preconditioner, preconditioner update and gradient preconditioning share some computations, but our implementations do not exploit this for favoring readability; the dummy variable $v$ is integrated out only if doing so reduces computations. Except for the Affine preconditioners, no native support of complex parameter optimization (you can define complex parameters as view of real ones in order to use other preconditioners). No line search is implemented for the conventional convex optimization setting.     

### Demos 

[Rosenbrock function minimization](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): see how simple to apply PSGD to convex and stochastic optimizations. Typically, we just set preconditioner_init_scale=None, and may need to tune lr_params and lr_preconditioner a little. 

[LeNet5](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): demonstration of PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation and comparison with Shampoo. 
 
[Delayed XOR](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this "naive" problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[Tensor rank decomposition benchmark](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all preconditioners on the tensor rank decomposition problem. PSGD performs better than BFGS.     

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large-scale ($6.2$ M coefficients) logistic regression problem. PSGD performs better than LM-BFGS, "the algorithm of choice" for logistic regression.  

[Preconditioner fitting on Lie groups](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py): see how multiplicative updates work on Lie groups for different types of preconditioners: ${\rm GL}(n, \mathbb{R})$, LRA and Affine with $Q=Q_2\otimes Q_1$. 

[Preconditioner estimation efficiency and numerical stability](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py): a playground to compare PSGD with BFGS and closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$. Eigenvalue decompositions required by the closed-form solution can be numerically unstable with single precisions, while PSGD is free of any numerically problematic operations like large matrix inverse, eigenvalue decompositions, etc.

[How PSGD generalizes so well](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py): This one serves as a good toy example illustrating it in the view of information theory. Starting from the same initial guesses, PSGD tends to find minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. Similarly, [this example](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) shows that PSGD also generalizes better than Shampoo. 

PSGD vs approximated closed-form solutions: [This example](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_shampoo_caspr.py) shows that most approximated closed-form solutions, e.g., KFAC, Shampoo, CASPR to name a few, are pretty rough in terms of accuracy, although they do work in certain circumstances. [Another example](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_vs_adafactor.py) compares Adafactor and PSGD with Lie group ${\rm diag}(q_2)\otimes {\rm diag}(q_1)$. There is no exact closed-form solution for $Q$ even with such a naively simple Lie group. The closed-form solution of Adafactor can be arbitrarily biased and generally doesn't work well out of the box.       

Wrapping as affine models: [this demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) on wrapping torch.nn.functional.conv2d as an affine Conv2d class, and [this one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class.

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Stochastic Hessian fittings on Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD. The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)
6) Other implementations: [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf) (outdated and unmaintained); [JAX](https://github.com/evanatyourservice/psgd_jax).
