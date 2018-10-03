### Pytorch implementation of preconditioned SGD (PSGD)
#### Overview
This package implements the Newton type and Fisher type preconditioned SGD methods [1, 2]. The Fisher type method applies to stochastic learning where Fisher metric can be well defined, while the Newton type method applies to a much wider range of applications. We have implemented dense, diagonal, sparse LU decomposition, Kronecker product, scaling-and-normalization and scaling-and-whitening preconditioners. Many optimization methods are closely related to certain specific realizations of our methods, e.g., Adam-->Fisher type + diagonal preconditioner + momentum, KFAC-->Fisher type + Kronecker product preconditioner, batch normalization-->scaling-and-normalization preconditioner, equilibrated SGD-->Newton type + diagonal preconditioner, etc. Please check [2] for further details on this package.       
#### About the code
*'hello_psgd.py'*: please try it first to see whether the code works for you. We verified it on Pytorch 0.4. 

*'preconditioned_stochastic_gradient_descent.py'*: it defines the preconditioners and preconditioned gradients we have developed. 

*'demo_psgd_....py'*: these files demonstrate the usage of the Newton type method along with different preconditioners. It is possible to combine preconditioning with momentum, and we do not show it here.

*'demo_fisher_type_psgd_scaw.py'*: it demonstrates the usage of Fisher type method. Of course, we can change the preconditioner, and combine this method with momentum.

*'rnn_add_problem_data_model_loss.py'*: it defines a simple RNN learning benchmark problem to test our demos.
#### Misc.
*Non-differentiable functions*: Non-differentiable functions, e.g., ReLU, max pooling, lead to vanishing/undefined/ill-conditioned Hessian and Fisher metric. PSGD with sparse preconditioners is empirically shown to work with such functions. Just be cautious with such models.  

*Vanishing or ill-conditioned Hessian*: Such Hessians lead to excessively large preconditioner. Preconditioned gradient clipping helps to stabilize the learning in this case (similar to gradient clipping in RNN training).

*No support for 2nd derivative*: It is not uncommon to find that certain function implementations do not support 2nd derivative. For the Newton type methods, we may use the numerical differentiation method to approximate the Hessian-vector product.

*Save wall time per iteration*: possibly the simplest way is to update the preconditioners less frequently. Generally no big performance loss since curvatures typically evolve slower than gradients.

#### References
[1] Preconditioned Stochastic Gradient Descent, https://arxiv.org/abs/1512.04202, 2015. 

[2] Learning Preconditioners on Lie Groups, https://arxiv.org/abs/1809.10232, 2018.
