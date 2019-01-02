### Pytorch implementation of preconditioned SGD (PSGD)
#### Overview
This package implements the Newton type and Fisher type preconditioned SGD methods [1, 2]. The Fisher type method applies to stochastic learning where Fisher metric can be well defined, while the Newton type method applies to a much wider range of applications. We have implemented dense, diagonal, sparse LU decomposition, Kronecker product, scaling-and-normalization and scaling-and-whitening preconditioners. Many optimization methods are closely related to certain specific realizations of our methods, e.g., Adam-->(empirical) Fisher type + diagonal preconditioner + momentum, KFAC-->Fisher type + Kronecker product preconditioner, batch normalization-->scaling-and-normalization preconditioner, equilibrated SGD-->Newton type + diagonal preconditioner, etc. Please check [2] for further details.       
#### About the code
*'hello_psgd.py'*: please try it first to see whether the code works for you. 

*'preconditioned_stochastic_gradient_descent.py'*: it defines the preconditioners and preconditioned gradients we have developed. 

*'demo_psgd_....py'*: these files demonstrate the usage of the Newton type method along with different preconditioners. It is possible to combine preconditioning with momentum, and we do not show it here.

*'demo_fisher_type_psgd_scaw.py'*: it demonstrates the usage of Fisher type method. Of course, we can change the preconditioner, and combine this method with momentum. We use the *empirical* Fisher in this example. Estimating the true Fisher will be more involved [3].

*'demo_LeNet5.py'*: training the classic LeNet5 model with Newton type Kronecker product preconditioners. 

*'rnn_add_problem_data_model_loss.py'*: it defines a simple RNN learning benchmark problem to test our demos.

*'mnist_autoencoder_data_model_loss.py'*: it defines an autoencoder benchmark problem for testing KFAC [5]. First order methods perform poorly on this one.  
#### A quick benchmark on the MNIST dataset with LeNet5
Folder ./LeNet5 provides the code to do a quick benchmark on the classic MNIST handwritten digit recognition task with the classic LeNet5. With ten runs for each method, one set of typical (mean, std) test classification error rate numbers looks like:

METHOD | (MEAN, STD)%
------------ | -------------
Momentum | (0.96, 0.07)
Adam | (0.88, 0.07)
KFAC | (0.84, 0.07)
Fish type preconditioner | (0.86, 0.05)
Newton type preconditioner | (0.74, 0.06)
#### References
[1] Preconditioned stochastic gradient descent, https://arxiv.org/abs/1512.04202, 2015.  
[2] Preconditioner on matrix Lie group for SGD, https://arxiv.org/abs/1809.10232, 2018.  
[3] J. Martens, New insights and perspectives on the natural gradient method, https://arxiv.org/abs/1412.1193, 2014.  
[4] Please check https://github.com/lixilinx/psgd_tf for more information and comparisons with different methods.  
[5] https://medium.com/@yaroslavvb/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0
