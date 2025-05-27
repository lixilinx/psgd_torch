## Pytorch implementation of PSGD 
The [old PSGD implementation](https://github.com/lixilinx/psgd_torch/blob/master/preconditioned_stochastic_gradient_descent.py) is deprecated. The [new PSGD implementation](https://github.com/lixilinx/psgd_torch/blob/master/psgd.py) is a superset of the old one, and further supports three more matmul-only/inverse-free methods for updating $Q$. Recommended choices for dQ are QUAD and QEQ.   

### An overview
PSGD (Preconditioned SGD) is a general purpose (mathematical and stochastic, convex and nonconvex) 2nd order optimizer. It reformulates a wide range of preconditioner estimation and Hessian fitting problems as a family of strongly convex Lie group optimization problems. 

Notations: $E_z[\ell(\theta, z)]$ or $\ell(\theta)$ the loss;  $g$ the (stochastic) gradient wrt $\theta$; $H$ the Hessian;  $h=Hv$ the Hessian-vector product (Hvp) with ${v\sim\mathcal{N}(0,I)}$; $P=Q^TQ$ the preconditioner applying on $g$; ${\rm tri}$ takes the upper or lower triangular part of a matrix; $\lVert \cdot \rVert$ takes spectral norm; superscripts $^T$, $^*$ and $^H$ for transpose, conjugate and Hermitian transpose, respectively.
<!---; $[I+A]_R\approx I + {\rm triu}(A) + {\rm triu}(A,1)$ for $\|\|A\|\| < 1$, where $[\cdot]_R$ keeps $R$ of a QR decomposition.--->

The PSGD theory has two orthogonal parts: criteria for preconditioner fitting and preconditioner fitting in Lie groups. 

#### Criteria for preconditioner fitting 

PSGD was originally designed for preconditioning the gradient such that metrics of the spaces of preconditioned gradient and parameters are matched, i.e., $E_{\delta \theta, z}[(P\delta g)(P\delta g)^T] = E_{\delta \theta, z}[\delta \theta \delta \theta^T]$, where $\delta$ denotes the perturbation operation and $P$ is symmetric positive definite (SPD). This leads to the original preconditioner fitting criterion $E_{\delta\theta, z}[\delta g^T P \delta g + \delta \theta^T P^{-1} \delta \theta]$ [ref](https://arxiv.org/abs/1512.04202). The finite-difference notation may not be common in machine learning (ML). But, note that PSGD was invented before popular automatic differentiation (AD) tools like Tensorflow. Manually calculating the Hvp was cubersome then. With AD, we can simply replace pair $(\delta \theta, \delta g)$ with $(v, h)$ to obtain the Newton-style preconditioner fitting criterion $E_{v, z}[h^T P h + v^T P^{-1} v]$. For the gradient/momentum whitening preconditioner, we just replace pair $(\delta \theta, \delta g)$ with $(v, g)$ to have criterion $E_{v, z}[g^T P g + v^T P^{-1} v]$ [ref](https://arxiv.org/abs/1809.10232), where $v$ is an auxiliary variable and can be optionally integrated out as it is indepedent of $g$. 

<!--For optimizing the negative-logarithm-likelihood (NLL) losses and losses with only sub-gradients, the gradient whitening preconditioner is recommended. The Newton-type preconditioner is good for minimizing any smooth losses.-->

#### Preconditioner fitting in Lie groups

The above preconditioner fitting criteria are always convex in the Euclidean space, the manifold of SPD matrices and the Lie groups. But, they are *strongly* convex only in the Lie groups [ref](https://arxiv.org/abs/2402.11858). The $Q$ here defines the coordinate transform $\vartheta=Q^{-T}\theta$ such that PSGD reduces to an SGD for $\vartheta$. Lie group is a natural tool for this purpose by preserving invariances like the coordinate orientations such that $Q$ is always invertible. Also, the multiplicative updates in Lie group avoid explicit matrix inverse. There are virtually endless choices for the group forms of $Q$, say the Kronecker product preconditioner [ref](https://arxiv.org/abs/1512.04202), the affine Lie group [ref](https://arxiv.org/abs/1809.10232), and the low rank approximation (LRA) group [ref](https://arxiv.org/abs/2211.04422). 

<!--Generally, the Kron preconditioner is recommended for faster convergence. If quality of the solution is of the main interest, the LRA preconditioner could outperform the Kron one given enough iterations.-->     
<!--
#### Tips for numerical stability 

*Regularization/Damping*. When the Hvp is used for preconditioner fitting, $P$ eventually converges to $H^{-1}$ at a local minimum. Adding a small $L2$ regularization term, $0.5 \lambda \theta^T \theta$, to the loss $\ell(\theta)$ can lower bound the Hessian as $H\succeq \lambda I$ such that $P\preceq I/\lambda$. For the gradient whitening preconditioner, $P$ eventually converges to $(E[gg^T])^{-0.5}$. Adding a tiny bit of Gaussian noise, $v \sim \mathcal{N}(0, \lambda I)$, to $g$ also can lower bound $E[gg^T]$ as $E[gg^T]\succeq \lambda I$ such that $P\preceq \lambda^{-0.5} I$. How much damping is needed mainly depends on the machine precisions ([demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_with_finite_precision_arithmetic.py)).

*Trust region*. We may need to clip or shrink the norm of the preconditioned gradient $Pg$ to limit $\lVert \Delta \theta \rVert$ with a given learning rate (lr) such that $P$ still is a reasonably good preconditioner at $\theta + \Delta \theta$. How much clipping is needed depends on the size of lr and the roughness of the landscape of $\ell(\theta)$. In ML, this practice is known as gradient clipping, and typically clipping $Pg$ element-wisely or layer-wisely.   --> 

#### Table I: Variations of preconditioner fitting criterion
  
| Criterion | Solution | Notes |
|--------------|------------------|-------|
|$h^TPh + v^TP^{-1}v$ | $Phh^TP = vv^T$ | Reduces to secant equation $Ph=v$ when  $v^Th>0$ (see quasi-Newton methods, e.g., BFGS).   | 
| $E_v[h^TPh + v^TP^{-1}v]$ | $P^{-2}=H^2$ | Reduces to Newton's method when  $H\succ 0$.  | 
| $E_{v,z}[g_z^TPg_z + v^TP^{-1}v]$ | $P^{-2}=E_z[g_zg_z^T]$ | $P^{-2}$ reduces to Fisher information matrix $F$ with per-sample gradient $g_z$ (see Gauss-Newton and natural gradient methods, e.g., KFAC). | 
| $\sum_t E_{v_t}[g_t^TPg_t + v_t^TP^{-1}v_t]$ | $P^{-2}=\sum_t g_t g_t^T$ | Relates to the AdaGrad family, e.g., Adam(W), RMSProp, Shampoo, $\ldots$.  |   

Note 1: $v$ can be a nuisance or an auxiliary variable in the last two criteria since it is independent of $g$ and can be integrated out as $E_{v\sim\mathcal{N}(0,I)}[v^TP^{-1}v]={\rm tr}(P^{-1})$, i.e., the Hutchinson's estimator.   

#### Table II: Lie group ($dQ=EQ$) preconditioners with storage and computation numbers for $\theta={\rm vec}(\Theta)$ with $\Theta\in\mathbb{R}^{m\times m}$ 

|Lie Group|Update of $Q$ ($0<\mu\le 2$)  |Storages | Computations |Class|
|---|---|---|---|---|
| ${\rm GL}(n, \mathbb{R})$  | $Q\leftarrow \left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \lVert Qh\rVert ^2 + \lVert Q^{-T}v\rVert^2  } \right)   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^4)$ |  DenseNewton   |
| Tri matrices | $Q\leftarrow {\rm tri}\left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \lVert Qh\rVert^2 + \lVert Q^{-T}v\rVert^2  } \right)   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^6)$ |  DenseNewton     |
| $Q={\rm diag}(q)$ | $q\leftarrow \left( 1 - \mu \frac{(qh)^2 - (v/q)^2}{  \max\left((qh)^2 + (v/q)^2\right)} \right)   q$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^2)$ | LRAWhiten/Newton  |
| ${\rm kron}(Q_2,Q_1)$  | $A=Q_1 {\rm uvec}(h) Q_2^H$, $B=Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1}$, $Q_1\leftarrow {\rm tri}\left( I - \mu \frac{AA^H-B^HB}{\lVert AA^H+B^HB \rVert} \right)   Q_1$, $Q_2\leftarrow {\rm tri}\left( I - \mu \frac{A^HA-BB^H}{\lVert A^HA+BB^H \rVert} \right)   Q_2$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ | KronWhiten/Newton  | 
| ${\rm kron}(Q_1,Q_2,\ldots)$ | $A_{ab\ldots}=(Q_1)_{a \alpha}(Q_2)\_{b \beta}\ldots ({\rm uvec}(h))\_{\alpha\beta\ldots}$, $B^\*\_{ab\ldots}=({\rm uvec}(v^*))\_{\alpha\beta\ldots} (Q_1^{-1})\_{\alpha a} (Q_2^{-1})\_{\beta b}\ldots$, $(Q_i)\_{ac}\leftarrow {\rm tri}\left( I\_{ab} - \mu \frac{A\_{\ldots a\ldots}A^\*\_{\ldots b\ldots}-B\_{\ldots a\ldots}B^\*\_{\ldots b\ldots}}{\lVert A\_{\ldots a\ldots}A^\*\_{\ldots b\ldots}+B\_{\ldots a\ldots}B^\*\_{\ldots b\ldots} \rVert} \right)   Q\_{bc}$  | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ | KronWhiten/Newton  | 
| $Q=(I+UV^T){\rm diag}(d)$, $U, V \in \mathbb{R}^{n\times r}$, $0\le r\ll n$ | $a=Qh$, $b=Q^{-T}v$, $Q\leftarrow Q-\mu{\rm diag}(a^2-b^2)Q/\max(a^2+b^2)$, $U\leftarrow U - \mu\frac{(aa^T-bb^T)V(I+V^TU)}{\lVert a\rVert \\, \lVert VV^Ta \rVert + \lVert b\rVert \\, \lVert VV^Tb\rVert }$, $V\leftarrow V - \mu\frac{ (I+VU^T)(aa^T-bb^T)U }{\lVert a\rVert \\, \lVert UU^Ta\rVert + \lVert b\rVert \\, \lVert UU^Tb\rVert}$ | $\mathcal{O}(rm^2)$ | $\mathcal{O}(rm^2)$ | LRAWhiten/Newton  |  
| ${\rm diag}(q_1)\otimes{\rm diag}(q_2)\otimes\ldots$ | same as kron | $\mathcal{O}(m)$ | $\mathcal{O}(m^2)$ | KronWhiten/Newton | 

Note 1: The other three inverse-free preconditioner update methods have similar forms and complexities. Please check [ref](https://arxiv.org/abs/2402.11858) for further details. 

Note 2: For the gradient/momentum whitening preconditioner, we simply replace pair $(v, h)$ with $(v, g)$, where $v$ is a dummy variable that can be optionally integrated out. 

<!--Taking the gradient whitening affine preconditioner as an example, we can integrate out $v\sim\mathcal{N}(0,I)$ as 

$\eqalign{
E_v[BB^H] &= E_v \left\\{Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1} Q_1^{-H} [{\rm uvec}(v)] Q_2^{-1}\right\\} ={\rm tr}\left(Q_1^{-1}Q_1^{-H}\right) \\, Q_2^{-H}Q_2^{-1}\cr
E_v[B^HB] &= {\rm tr}\left(Q_2^{-1}Q_2^{-H}\right) \\, Q_1^{-H}Q_1^{-1}
}$-->

<!--$E_{v\sim\mathcal{N}(0,I)}[BB^H] = E_v \left(Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1} Q_1^{-H} [{\rm uvec}(v)] Q_2^{-1}\right) ={\rm tr}\left(Q_2^{-1}Q_2^{-H}\right) \\, Q_1^{-H}Q_1^{-1}$,  $\\;$ $E_v[B^HB] = {\rm tr}\left(Q_1^{-1}Q_1^{-H}\right) \\, Q_2^{-H}Q_2^{-1}$--->  

<!--The default behavior is to keep $v$ as an auxiliary variable. For class *Kron*, setting $v$ to None will integrate it out. -->

#### Hessian fitting accuracy   

[This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) generates the following plot showing the typical behaviors of different Hessian fitting methods. 

* With a static and noise-free Hessian-vector product model, both BFGS and PSGD converge linearly to the optimal preconditioner while closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ only converges sublinearly with rate $\mathcal{O}(1/t)$.
* With a static additive noisy Hessian-vector model $h=Hv+\epsilon$, BFGS diverges easily. With a constant step size $\mu$, the steady-state fitting errors of PSGD are proportional  to $\mu$. 
* With a time-varying Hessian $H_{t+1}=H_t + uu^T$ and $u\sim\mathcal{U}(0,1)$, PSGD locks onto good preconditioner estimations quicker than BFGS without a divergence stage. The closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ is not good at tracking due to its sublinear rate of convergence.       

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=90% height=90%>

#### Implementation details 
Optimizers with the criteria in Table I and preconditioner forms in Table II are wrapped into classes *KronWhiten/Newton*, *LRAWhiten/Newton* and *DenseNewton* for easy use.     

Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more dynamic behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivative is unavailable. The closure should return a loss or an iterator with its first element as the loss.     
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding an $L2$ regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementations. <!--We recommend to randomize the regularization term, e.g., replacing the $L2$ one for a parameter $p$, say $0.5 \lambda \cdot {\rm sum}(p^2)$, with ${\rm rand}() \cdot \lambda\cdot {\rm sum}(p^2)$.--> 

A few more details. The Hessian-vector products are calculated as a vector-jacobian-product (vjp), i.e., ${\rm autograd.grad}(g, \theta, v)$ in torch, maybe not always the most efficient way for a specific problem. Except for the Kronecker product preconditioners, no native support of complex parameter optimization (you can define complex parameters as view of real ones in order to use other preconditioners). <!--No line search is implemented for the conventional convex optimization setting.-->     

### Demos 

There are plenty of demos: [Rosenbrock function minimization](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py), [vision transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/vit.py), [generative pre-trained transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/gpt2.py), [logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py), [tensor rank decomposition](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py), etc.. For this tiny [vision transformer demo](https://github.com/lixilinx/psgd_torch/blob/master/misc/vit.py), the following results show that all the four PSGD-Kron-gradient-whitening preconditioners can improve the convergence a lot compared with Adam(W).      

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/vit_adam_vs_psgd.svg" width=70% height=70%>

The inverse-free versions of PSGD also work well with half precision. Please check this [GPT2](https://github.com/lixilinx/psgd_torch/blob/master/misc/gpt2.py) example for reproducing the following results.

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/gpt2_adamw_vs_psgd.svg" width=70% height=70%>

<!--
[Rosenbrock function](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): see how simple to apply PSGD to convex and stochastic optimizations. The most important three settings are: preconditioner_init_scale (unnormalized), lr_params (normalized) and lr_preconditioner (normalized). 

[LeNet5 CNN](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation and comparison with Shampoo (PSGD generalizes better). 

[Vision transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/vit.py): CIFAR image recognition with a tiny transformer. PSGD converges faster and generalizes better than Adam(W). Check [here](https://drive.google.com/file/d/1nOnl8MW2OdWGriyR1rn3DqEJ8IqMwoG4/view?usp=drive_link) for sample results.

[Generative pre-trained transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/gpt2.py): A tiny GPT model for the WikiText-103 Dataset. PSGD also converges faster and generalizes better than Adam(W). Check [here](https://drive.google.com/file/d/176jqghbsOWtJPMSVpAx1JlN5pVMRCd2O/view?usp=drive_link) for sample results.   
 
[Delayed XOR with RNN](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Most optimizers can't crack this fundamental problem with either LSTM or the vanilla RNN, while PSGD can, with either LSTM or simple RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large-scale logistic regression problem. PSGD outperforms LM-BFGS, "the algorithm of choice" for this type of problem.  

[Tensor rank decomposition](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all preconditioners on the tensor rank decomposition problem. It's a classic math optimization problem and PSGD outperforms BFGS again.     

[PSGD vs approximated closed-form solutions](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_shampoo_caspr.py): this example show that most closed-form solutions, e.g., KFAC, Shampoo, CASPR, are approximate even for $H$ has the assumed form $H_2\otimes H_1$. For the simplest Kron preconditioner $Q={\rm diag}(q_2)\otimes {\rm diag}(q_1)$, [this example](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_vs_adafactor.py) shows that Adafactor can be biased too. These biased solutions generally need grafting (onto Adam or RMSProp) to work properly, while PSGD doesn't.    

[Preconditioner fitting in Lie groups](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py): see how multiplicative updates work in Lie groups for different types of preconditioners: ${\rm GL}(n, \mathbb{R})$, LRA and Affine. [One more example](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_kron_verification.py) for Kron.  

[Preconditioner estimation efficiency and numerical stability](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py): a playground to compare PSGD with BFGS and closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$. Eigenvalue decompositions required by the closed-form solution can be numerically unstable with single precisions, while PSGD is free of any potentially numerically problematic operations like large matrix inverse, eigenvalue decompositions, etc.

[How PSGD generalizes so well](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py): We know SGD generalizes. This one serves as a good toy example illustrating it in the view of information theory. Starting from the same initial guesses, PSGD tends to find minima with smaller train cross entropy and flatter Hessians than Adam. Thus shorter total description lengths for the train data and model parameters. See [sample results](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.svg). Similarly, [this example](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) shows that PSGD also generalizes better than Shampoo. 

[Wrapping as affine models](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py): this demo shows how to wrap torch.nn.functional.conv2d as an affine Conv2d class by putting weights and bias together. [Another one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class. It's tedious and also maybe unnecessary as the Kron preconditioners natively support tensors of any shape. Still, reformulating our model as a list of affine transforms can make the best use of Kron preconditioners and typically improves performance.   
-->

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting criteria and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on affine Lie group preconditioners, including feature normalization or whitening (per batch or layer) as special affine preconditioners. Use PSGD for gradient whitening.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Stochastic Hessian fittings with Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Properties of PSGD, also a good summary of PSGD. The Hessian fitting problem is shown to be strongly convex in ${\rm GL}(n, \mathbb{R})$ under certain mild assumptions.)
5) Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)
6) There are a few more efficient and specialized PSGD implementations: Evan's [JAX](https://github.com/evanatyourservice/psgd_jax) and [Torch](https://github.com/evanatyourservice/kron_torch) versions, Lucas' [Heavyball](https://github.com/ClashLuke/HeavyBall). Also my outdated and unmaintained Tensorflow code: [TF 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TF 2.x](https://github.com/lixilinx/psgd_tf). 
