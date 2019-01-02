Benchmarking the performance of the following methods on the MNIST dataset:

Momentum
Adam
KFAC
Fisher type preconditioner (square root version)
Newton type preconditioner

For each method, we train it with ten epochs, anneal its learning rate from lr to 0.01*lr, 
and tune its hyperparameters to optimize its test classification error rate.
