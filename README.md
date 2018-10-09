### Pytorch implementation of preconditioned SGD (PSGD)
#### Overview
This package implements the Newton type and Fisher type preconditioned SGD methods [1, 2]. The Fisher type method applies to stochastic learning where Fisher metric can be well defined, while the Newton type method applies to a much wider range of applications. We have implemented dense, diagonal, sparse LU decomposition, Kronecker product, scaling-and-normalization and scaling-and-whitening preconditioners. Many optimization methods are closely related to certain specific realizations of our methods, e.g., Adam-->(empirical) Fisher type + diagonal preconditioner + momentum, KFAC-->Fisher type + Kronecker product preconditioner, batch normalization-->scaling-and-normalization preconditioner, equilibrated SGD-->Newton type + diagonal preconditioner, etc. Please check [2] for further details.       
#### About the code
*'hello_psgd.py'*: please try it first to see whether the code works for you. We verified it on Pytorch 0.4. 

*'preconditioned_stochastic_gradient_descent.py'*: it defines the preconditioners and preconditioned gradients we have developed. 

*'demo_psgd_....py'*: these files demonstrate the usage of the Newton type method along with different preconditioners. It is possible to combine preconditioning with momentum, and we do not show it here.

*'demo_fisher_type_psgd_scaw.py'*: it demonstrates the usage of Fisher type method. Of course, we can change the preconditioner, and combine this method with momentum. We use the *empirical* Fisher in this example. Estimating the true Fisher will be more involved [3].    

*'rnn_add_problem_data_model_loss.py'*: it defines a simple RNN learning benchmark problem to test our demos.
#### References
[1] Preconditioned stochastic gradient descent, https://arxiv.org/abs/1512.04202, 2015.  
[2] Learning preconditioners on Lie groups, https://arxiv.org/abs/1809.10232, 2018.  
[3] J. Martens, New insights and perspectives on the natural gradient method, https://arxiv.org/pdf/1503.05671.pdf, 2014.  
[4] Please check https://github.com/lixilinx/psgd_tf for more information.
