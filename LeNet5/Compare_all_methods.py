import matplotlib.pyplot as plt

for _ in range(10):
    exec(open('LeNet5_momentum.py').read())
    exec(open('LeNet5_adam.py').read())
    exec(open('LeNet5_kfac.py').read())
    exec(open('LeNet5_fisher_kron.py').read())
    exec(open('LeNet5_newton_kron.py').read())
    plt.subplot(2,1,1)
    plt.legend(['Momentum', 'Adam', 'KFAC', 'Fisher type preconditioner', 'Newton type preconditioner'], loc='best')

plt.show()