## Pytorch implementation of PSGD 
*Major recent updates*: added gradient whitening preconditioners to all classes; wrapped affine preconditioners as a class; preconditioner fitting step size normalization with the 2nd derivative info; added Kron class as a superset of Affine (natively support tensors of any dims).
### An overview
PSGD (Preconditioned SGD) is a general purpose (mathematical and stochastic, convex and nonconvex) 2nd order optimizer. It reformulates a wide range of preconditioner estimation and Hessian fitting problems as a family of strongly convex Lie group optimization problems. 

Notations: $E_z[\ell(\theta, z)]$ or $\ell(\theta)$ the loss;  $g$ the (stochastic) gradient wrt $\theta$; $H$ the Hessian;  $h=Hv$ the Hessian-vector product with ${v\sim\mathcal{N}(0,I)}$; $P=Q^TQ$ the preconditioner applying on $g$; ${\rm tri}$ takes the upper or lower triangular part of a matrix; $\lVert \cdot \rVert$ takes spectral norm; superscripts $^T$, $^*$ and $^H$ for transpose, conjugate and Hermitian transpose, respectively.
<!---; $[I+A]_R\approx I + {\rm triu}(A) + {\rm triu}(A,1)$ for $\|\|A\|\| < 1$, where $[\cdot]_R$ keeps $R$ of a QR decomposition.--->

#### Table I: Variations of preconditioner fitting criterion
  
| Criterion | Solution | Notes |
|--------------|------------------|-------|
|$h^TPh + v^TP^{-1}v$ | $Phh^TP = vv^T$ | Reduces to secant equation $Ph=v$ when  $v^Th>0$ (Quasi-Newton methods, e.g., BFGS); set preconditioner_type="Newton" for this implementation.   | 
| $E_v[h^TPh + v^TP^{-1}v]$ | $P^{-2}=H^2$ | Reduces to Newton's method when  $H\succ 0$; set preconditioner_type="Newton" for this implementation.  | 
| $E_{v,z}[g^TPg + v^TP^{-1}v]$ | $P^{-2}=E_z[gg^T]$ | $P^{-2}$ reduces to Fisher information matrix $F$ with sample-wise gradient $g$ (Gauss-Newton and natural gradient family, e.g., KFAC); *no class implementation* (simply applying $P$ twice to get $F^{-1}$). | 
| $\sum_t E_{v_t}[g_t^TPg_t + v_t^TP^{-1}v_t]$ | $P^{-2}=\sum_t g_t g_t^T$ | Reduces to AdaGrad family, e.g., Adam(W), RMSProp, Shampoo, $\ldots$; set preconditioner_type="whitening" for this implementation.  |   

Note that $v$ can be a nuisance or an auxiliary variable in the last two criteria since it is independent of $g$ and can be integrated out as $E_{v\sim\mathcal{N}(0,I)}[v^TP^{-1}v]={\rm tr}(P^{-1})$.   

#### Table II: Implemented Lie group preconditioners with storage and computation numbers for $\theta={\rm vec}(\Theta)$ and $\Theta\in\mathbb{R}^{m\times m}$ 

|Lie Group|Update of $Q$ ($0<\mu\le 2$)  |Storages | Computations |Notes|
|---|---|---|---|---|
| ${\rm GL}(n, \mathbb{R})$  | $Q\leftarrow \left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \lVert Qh\rVert ^2 + \lVert Q^{-T}v\rVert^2  } \right)   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^4)$ | See class *Newton*; set keep_invQ=True to calculate $Q^{-1}$ recursively via Woodbury formula.   |
| Triangular matrices | $Q\leftarrow {\rm tri}\left( I - \mu \frac{Qhh^TQ^T - Q^{-T}vv^TQ^{-1}}{ \lVert Qh\rVert^2 + \lVert Q^{-T}v\rVert^2  } \right)   Q$ | $\mathcal{O}(m^4)$ | $\mathcal{O}(m^6)$ | See class *Newton*; set keep_invQ=False to make $Q$ triangular and calculate $Q^{-T}v$ with backward substitution.     |
| Diagonal matrices, $Q={\rm diag}(q)$ | $q\leftarrow \left( 1 - \mu \frac{(qh)^2 - (v/q)^2}{  \max\left((qh)^2 + (v/q)^2\right)} \right)   q$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^2)$ | See class either *LRA* with rank_of_approximation=0 or *XMat* for implementations.  |
| $Q_1$ and $Q_2$ are triangular, $Q=Q_2\otimes Q_1$  | $A=Q_1 {\rm uvec}(h) Q_2^H$, $B=Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1}$, $Q_1\leftarrow {\rm tri}\left( I - \mu \frac{AA^H-B^HB}{\lVert AA^H+B^HB \rVert} \right)   Q_1$, $Q_2\leftarrow {\rm tri}\left( I - \mu \frac{A^HA-BB^H}{\lVert A^HA+BB^H \rVert} \right)   Q_2$ | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ | See class *Affine* for implementations; deprecated, and upgraded to class *Kron*.  | 
| $Q$ defines einsum $T\_{ab\ldots} \leftarrow (Q_1)\_{a\alpha}(Q_2)_{b\beta}\ldots T\_{\alpha\beta\ldots}$ | $A_{ab\ldots}=(Q_1)_{a \alpha}(Q_2)\_{b \beta}\ldots ({\rm uvec}(h))\_{\alpha\beta\ldots}$, $B^\*\_{ab\ldots}=({\rm uvec}(v^*))\_{\alpha\beta\ldots} (Q_1^{-1})\_{\alpha a} (Q_2^{-1})\_{\beta b}\ldots$, $(Q_i)\_{ac}\leftarrow {\rm tri}\left( I\_{ab} - \mu \frac{A\_{\ldots a\ldots}A^\*\_{\ldots b\ldots}-B\_{\ldots a\ldots}B^\*\_{\ldots b\ldots}}{\lVert A\_{\ldots a\ldots}A^\*\_{\ldots b\ldots}+B\_{\ldots a\ldots}B^\*\_{\ldots b\ldots} \rVert} \right)   Q\_{bc}$  | $\mathcal{O}(m^2)$ | $\mathcal{O}(m^3)$ | See class *Kron* for implementations (a superset of *Affine*).  | 
| $Q=(I+UV^T){\rm diag}(d)$ | $a=Qh$, $b=Q^{-T}v$, $d\leftarrow \left(1 - \mu\frac{(Q^Ta)h-v(Q^{-1}b)}{\max\sqrt{\left((Q^Ta)^2 + v^2\right)\left(h^2 + (Q^{-1}b)^2\right)}}\right)d$, $U\leftarrow U - \mu\frac{(aa^T-bb^T)V(I+V^TU)}{\lVert a\rVert \\, \lVert VV^Ta \rVert + \lVert b\rVert \\, \lVert VV^Tb\rVert }$, $V\leftarrow V - \mu\frac{ (I+VU^T)(aa^T-bb^T)U }{\lVert a\rVert \\, \lVert UU^Ta\rVert + \lVert b\rVert \\, \lVert UU^Tb\rVert}$ | $\mathcal{O}(rm^2)$ | $\mathcal{O}(rm^2)$ | See class *LRA* for implementations; typically $0\le r\ll n$ with $U, V \in \mathbb{R}^{n\times r}$. Update either $U$ or $V$, not both, per step.  |  
| Scaling or normalization | Similar to *Kron*, but $Q_i$ only for scaling or normalization. | $\mathcal{O}(m)$ | $\mathcal{O}(m^2)$ | With *Kron*, small preconditioner_max_size or preconditioner_max_skew makes large $Q_i$ diagonal. | 

For AdaGrad like gradient whitening preconditioner, we simply replace pair $(v, h)$ with $(v, g)$, where $v$ is a dummy variable that can be optionally integrated out. Taking the gradient whitening affine preconditioner as an example, we can integrate out $v\sim\mathcal{N}(0,I)$ as 

$\eqalign{
E_v[BB^H] &= E_v \left\\{Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1} Q_1^{-H} [{\rm uvec}(v)] Q_2^{-1}\right\\} ={\rm tr}\left(Q_1^{-1}Q_1^{-H}\right) \\, Q_2^{-H}Q_2^{-1}\cr
E_v[B^HB] &= {\rm tr}\left(Q_2^{-1}Q_2^{-H}\right) \\, Q_1^{-H}Q_1^{-1}
}$

<!--$E_{v\sim\mathcal{N}(0,I)}[BB^H] = E_v \left(Q_2^{-H} [{\rm uvec}(v)]^H Q_1^{-1} Q_1^{-H} [{\rm uvec}(v)] Q_2^{-1}\right) ={\rm tr}\left(Q_2^{-1}Q_2^{-H}\right) \\, Q_1^{-H}Q_1^{-1}$,  $\\;$ $E_v[B^HB] = {\rm tr}\left(Q_1^{-1}Q_1^{-H}\right) \\, Q_2^{-H}Q_2^{-1}$--->  

Yet, keeping $v$ as an auxiliary variable simplifies the calculations most of the time.  

#### Preconditioner fitting accuracy   

[This script](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py) generates the following plot showing the typical behaviors of different preconditioner fitting methods. 

* With a static and noise-free Hessian-vector product model, both BFGS and PSGD converge linearly to the optimal preconditioner while closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ only converges sublinearly with rate $\mathcal{O}(1/t)$.
* With a static additive noisy Hessian-vector model $h=Hv+\epsilon$, BFGS diverges easily. With a constant step size $\mu$, the steady-state fitting errors of PSGD are proportional  to $\mu$. 
* With a time-varying Hessian $H_{t+1}=H_t + uu^T$ and $u\sim\mathcal{U}(0,1)$, PSGD locks onto good preconditioner estimations quicker than BFGS, also no divergence before convergence. The closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$ is not good at tracking due to its sublinear convergence.       

<img src="https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.svg" width=90% height=90%>

#### Implementation details 
Optimizers with the criteria in Table I and preconditioner forms in Table II are wrapped into classes *XMat*, *LRA* (or *UVd*), *Newton*, *Affine* and *Kron* for easy use. The Affine family are obsolete now as Kron is a superset of it.  

Three main differences from torch.optim.SGD: 
1) The loss to be minimized is passed through as a closure to the optimizer to support more complicated behaviors, notably, Hessian-vector product approximation with finite difference method when the 2nd order derivatives are unavailable.   
2) Momentum here is the moving average of gradient so that its setting is decoupled from the learning rate, which is always normalized in PSGD. 
3) As any other regularizations, (coupled) weight decay should be explicitly realized by adding $L2$ regularization to the loss. Similarly, decoupled weight decay is not included inside the PSGD implementations. We recommend to randomize the regularization term, e.g., replacing the $L2$ one for a parameter $p$,  $0.5 \lambda \cdot {\rm sum}(p^2)$, with ${\rm rand}() \cdot \lambda\cdot {\rm sum}(p^2)$. 

A few more details. The Hessian-vector products are calculated as a vector-jacobian-product (vjp), i.e., ${\rm autograd.grad}(g, \theta, v)$ in torch, maybe not always the most efficient way for a specific problem. For gradient whitening preconditioner, preconditioner update and gradient preconditioning share some computations, but our implementations do not exploit this for favoring readability. Except for the Affine and Kron preconditioners, no native support of complex parameter optimization (you can define complex parameters as view of real ones in order to use other preconditioners). No line search is implemented for the conventional convex optimization setting.     

### Demos 

[Rosenbrock function](https://github.com/lixilinx/psgd_torch/blob/master/hello_psgd.py): see how simple to apply PSGD to convex and stochastic optimizations. The most important three settings are: preconditioner_init_scale (unnormalized), lr_params (normalized) and lr_preconditioner (normalized). 

[LeNet5 CNN](https://github.com/lixilinx/psgd_torch/blob/master/mnist_with_lenet5.py): PSGD on convolutional neural network training with the classic LeNet5 for MNIST digits recognition. Also see [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) for another implementation and comparison with Shampoo (PSGD generalizes significantly better). 

[Vision transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/vit.py): CIFAR image recognition with a tiny transformer. PSGD converges significantly faster and generalizes better than Adam(W). Check [here](https://drive.google.com/file/d/1nOnl8MW2OdWGriyR1rn3DqEJ8IqMwoG4/view?usp=drive_link) for sample results.

[Generative pre-trained transformer](https://github.com/lixilinx/psgd_torch/blob/master/misc/gpt2.py): A tiny GPT model for the WikiText-103 Dataset. PSGD also converges faster and generalizes better than Adam(W). Check [here](https://drive.google.com/file/d/176jqghbsOWtJPMSVpAx1JlN5pVMRCd2O/view?usp=drive_link) for sample results.   
 
[Delayed XOR with RNN](https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py): demonstration of PSGD on gated recurrent neural network (RNN) learning with the delayed XOR problem proposed in the [LSTM paper](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory). Note that neither LSTM nor the vanilla RNN can solve this "naive" problem with most optimizers, while PSGD can, with either the LSTM or the vanilla RNN (also see [this](https://github.com/lixilinx/psgd_torch/blob/master/rnn_xor_problem_general_purpose_preconditioner.py) and [this](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) with simple RNNs).

[Logistic regression](https://github.com/lixilinx/psgd_torch/blob/master/misc/mnist_logistic_regression.py): a large-scale ($6.2$ M coefficients) logistic regression problem. PSGD outperforms LM-BFGS, "the algorithm of choice" for it.  

[Tensor rank decomposition](https://github.com/lixilinx/psgd_torch/blob/master/demo_usage_of_all_preconditioners.py): demonstrate the usage of all preconditioners on the tensor rank decomposition problem. PSGD performs better than BFGS.     

[PSGD vs approximated closed-form solutions](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_shampoo_caspr.py): this example show that most closed-form solutions for preconditioners, e.g., KFAC, Shampoo, and CASPR, are approximate and inaccurate. Even for the simplest affine preconditioner $Q={\rm diag}(q_2)\otimes {\rm diag}(q_1)$, [this example](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_vs_adafactor.py) shows that the closed-form solution of Adafactor could be poor. These biased solutions may need grafting to work properly, while PSGD does not.    

[Preconditioner fitting on Lie groups](https://github.com/lixilinx/psgd_torch/blob/master/misc/preconditioner_fitting_rule_verification.py): see how multiplicative updates work on Lie groups for different types of preconditioners: ${\rm GL}(n, \mathbb{R})$, LRA and Affine with $Q=Q_2\otimes Q_1$. 

[Preconditioner estimation efficiency and numerical stability](https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_numerical_stability.py): a playground to compare PSGD with BFGS and closed-form solution $P=\left(E[hh^T]\right)^{-0.5}$. Eigenvalue decompositions required by the closed-form solution can be numerically unstable with single precisions, while PSGD is free of any numerically problematic operations like large matrix inverse, eigenvalue decompositions, etc.

[How PSGD generalizes so well](https://github.com/lixilinx/psgd_torch/blob/master/misc/how_psgd_generalize.py): This one serves as a good toy example illustrating it in the view of information theory. Starting from the same initial guesses, PSGD tends to find minima with smaller train cross entropy and flatter Hessians than Adam, i.e., shorter description lengths (DL) for train image-label pairs and model parameters. Similarly, [this example](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py) shows that PSGD also generalizes better than Shampoo. 

[Wrapping as affine models](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_F_conv2d.py): this demo shows how to wrap torch.nn.functional.conv2d as an affine Conv2d class. Almost all the neural network models are built on affine transforms, e.g., [another one](https://github.com/lixilinx/psgd_torch/blob/master/misc/affine_wrapping_VF_rnn_tanh.py) on wrapping torch._VF.rnn_tanh as an affine RNN class. Now, the Kron preconditioners can natively support tensors of any dims without reshaping.  

### Resources
1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie groups.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. I also have prepared [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Stochastic Hessian fittings on Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD. The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)
6) Other implementations: [Tensorflow 1.x](https://github.com/lixilinx/psgd_tf/releases/tag/1.3) and [TensorFlow 2.x](https://github.com/lixilinx/psgd_tf) (outdated and unmaintained); [JAX](https://github.com/evanatyourservice/psgd_jax).
