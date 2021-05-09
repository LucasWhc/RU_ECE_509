import matplotlib.pyplot as plt
import numpy as np


x = np.array([5.0, 5.0])
max_iter = 50
f = lambda x: (np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1))
dfx1 = lambda x: (np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1))
dfx2 = lambda x: (3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(-3 * x[1] - 0.1 + x[0]))
xl = np.linspace(1, max_iter, max_iter)
yl = []


def get_step(x0, t, alpha, beta):
    """"
    x0: array stored data x1, x2
    dfx1: gradient for x1
    dfx2: gradient for x2
    t: step size
    """
    while (f(x0) - (f(x0 - t * np.array([dfx1(x0), dfx2(x0)])) + alpha * t * np.dot(np.array([dfx1(x0), dfx2(x0)]),
                                                                                    np.array([dfx1(x0),
                                                                                              dfx2(x0)])))) < 0:
        t *= beta
    return t


def gradient_descent():
    for iter in range(max_iter):
        alpha = 0.1
        beta = 0.6
        t = 1
        val = f(x)
        print("Iteration: ", iter, ":  x1 = ", x[0], ", x2 = ", x[1], ", f(x1,x2) = ", val)
        yl.append(np.log(val))
        t = get_step(x, t, alpha, beta)
        print("optimal step size: ", t)
        x[0] = x[0] - dfx1(x) * t
        x[1] = x[1] - dfx2(x) * t
    plt.xlabel('K')
    plt.ylabel('log(f(X))')
    plt.plot(xl, yl)
    plt.savefig('4_1.jpg')
    plt.show()


if __name__ == "__main__":
    gradient_descent()
