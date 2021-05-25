"""
Code to compare behavior of isotropic Gaussians optimized with respect to
different divergences.

Adapted from:
https://github.com/lucastheis/model-evaluation/blob/master/code/experiments/comparison.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import os
import seaborn as sns
import sys
import theano as th
import theano.sandbox.linalg as tl
import theano.tensor as tt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time

# from notebooks.utils import PlotParams

sys.path.append('./code')
mpl.use('Agg')


def normal(X, m, C):
    """
    Evaluates the density of a normal distribution.

    @type  X: C{TensorVariable}
    @param X: matrix storing data points column-wise

    @type  m: C{ndarray}/C{TensorVariable}
    @param m: column vector representing the mean of the Gaussian

    @type  C: C{ndarray}/C{TensorVariable}
    @param C: covariance matrix

    @rtype: C{TensorVariable}
    @return: density of a Gaussian distribution evaluated at C{X}
    """

    Z = X - m

    return tt.exp(
        - tt.sum(Z * tt.dot(tl.matrix_inverse(C), Z), 0) / 2.
        - tt.log(tl.det(C)) / 2.
        - m.size / 2. * np.log(2. * np.pi))


def mogaussian(D=2, K=10, N=100000, seed=2, D_max=100):
    """
    Creates a random mixture of Gaussians and corresponding samples.

    @rtype: C{tuple}
    @return: a function representing the density and samples
    """

    nr.seed(seed)

    # mixture weights
    p = nr.dirichlet([.5] * K)

    # variances
    v = 1. / np.square(nr.rand(K) + 1.)

    # means; D_max makes sure that data only depends on seed and not on D
    m = nr.randn(D_max, K) * 1.5
    m = m[:D]
    # m is a numpy array which is normally distributed with N(0, (1.5**2)) and has shape (2, 10)

    # density function
    X = tt.dmatrix('X')
    C = [np.eye(D) * _ for _ in v]

    def log_p(X):
        """
        @type  X: C{ndarray}/C{TensorVariable}
        @param X: data points stored column-wise

        @rtype: C{ndarray}/C{TensorVariable}
        """

        if isinstance(X, tt.TensorVariable):
            return tt.log(tt.sum([p[i] * normal(X, m[:, [i]], C[i]) for i in range(len(p))], 0))
        else:
            if log_p.f is None:
                Y = tt.dmatrix('Y')
                log_p.f = th.function([Y], log_p(Y))
            return log_p.f(X)
    log_p.f = None

    def nonlog_p(X):
        """
        @type  X: C{ndarray}/C{TensorVariable}
        @param X: data points stored column-wise

        @rtype: C{ndarray}/C{TensorVariable}
        """

        if isinstance(X, tt.TensorVariable):
            return tt.sum([p[i] * normal(X, m[:, [i]], C[i]) for i in range(len(p))], 0)
        else:
            if nonlog_p.f is None:
                Y = tt.dmatrix('Y')
                nonlog_p.f = th.function([Y], nonlog_p(Y))
            return nonlog_p.f(X)
    nonlog_p.f = None

    # sample data
    M = nr.multinomial(N, p)
    data = np.hstack(nr.randn(D, M[i]) * np.sqrt(v[i]) + m[:, [i]] for i in range(len(p)))
    data = data[:, nr.permutation(N)]
    return nonlog_p, log_p, data


def ravel(params):
    """
    Combine parameters into a long one-dimensional array.

    @type  params: C{list}
    @param params: list of shared variables

    @rtype: C{ndarray}
    """
    return np.hstack(p.get_value().ravel() for p in params)


def unravel(params, x):
    """
    Extract parameters from an array and insert into shared variables.

    @type  params: C{list}
    @param params: list of shared variables

    @type  x: C{ndarray}
    @param x: parameter values
    """
    x = x.ravel()
    for param in params:
        param.set_value(x[:param.size.eval()].reshape(param.shape.eval()))
        x = x[param.size.eval():]


def plot(log_q, data, xmin=-5, xmax=7, ymin=-5, ymax=7):
    """
    Visualize density (as contour plot) and data samples (as histogram).
    """

    if isinstance(log_q, tuple) or isinstance(log_q, list):
        A, b = log_q
        X = tt.dmatrix('X')
        log_q = th.function([X], normal(X, b, np.dot(A, A.T)))

    # evaluate density on a grid
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    zz = np.exp(log_q(np.asarray([xx.ravel(), yy.ravel()])).reshape(xx.shape))

    hh, x, y = np.histogram2d(data[0], data[1], 80, range=[(xmin, xmax), (ymin, ymax)])

    sns.set_style('whitegrid')
    sns.set_style('ticks')
    plt.figure(figsize=(10, 10), dpi=300)
    # plt.imshow(hh.T[::-1], extent=[x[0], x[-1], y[0], y[-1]],
    #   interpolation='nearest', cmap='YlGnBu_r')
    # plt.contour(xx, yy, zz, 7, colors='w', alpha=.7)
    plt.scatter(data[0], data[1], color='k', marker='.', alpha=0.05)
    plt.contour(xx, yy, zz, 5, linewidths=2)
    plt.axis('equal')
    plt.axis([x[0], x[-1], y[0], y[-1]])
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.gcf().tight_layout()


def fit_mmd(data):
    """
    Fit isotropic Gaussian by minimizing maximum mean discrepancy.

    B{References:}
        - A. Gretton et al., I{A Kernel Method for the Two-Sample-Problem}, NIPS, 2007
        - Y. Li et al., I{Generative Moment Matching Networks}, ICML, 2015
    """

    def gaussian_kernel(x, y, sigma=1.):
        return tt.exp(-tt.sum(tt.square(x - y)) / sigma**2)

    def mixed_kernel(x, y, sigma=[.5, 1., 2., 4., 8.]):
        return tt.sum([gaussian_kernel(x, y, s) for s in sigma])

    def gram_matrix(X, Y, kernel):
        M = X.shape[0]
        N = Y.shape[0]

        G, _ = th.scan(
            fn=lambda k: kernel(X[k // N], Y[k % N]),
            sequences=[tt.arange(M * N)])

        return G.reshape([M, N])

    # hiddens
    Z = tt.dmatrix('Z')

    # parameters
    b = th.shared(np.mean(data, 1)[None], broadcastable=[True, False])
    A = th.shared(np.std(data - b.get_value().T))

    # model samples
    X = Z * A + b

    # data
    Y = tt.dmatrix('Y')
    M = X.shape[0]
    N = Y.shape[0]

    Kyy = gram_matrix(Y, Y, mixed_kernel)
    Kxy = gram_matrix(X, Y, mixed_kernel)
    Kxx = gram_matrix(X, X, mixed_kernel)

    MMDsq = tt.sum(Kxx) / M**2 - 2. / (N * M) * tt.sum(Kxy) + tt.sum(Kyy) / N**2
    MMD = tt.sqrt(MMDsq)

    f = th.function([Z, Y], [MMD, tt.grad(MMD, A), tt.grad(MMD, b)])

    # batch size, momentum, learning rate schedule
    B = 100
    mm = 0.8
    kappa = .7
    tau = 1.

    values = []

    try:
        for t in range(0, data.shape[1], B):
            if t % 10000 == 0:
                # reset momentum
                dA = 0.
                db = 0.

            Z = nr.randn(B, data.shape[0])
            Y = data.T[t:t + B]

            lr = np.power(tau + (t + B) / B, -kappa)

            v, gA, gb = f(Z, Y)
            dA = mm * dA - lr * gA
            db = mm * db - lr * gb

            values.append(v)

            A.set_value(A.get_value() + dA)
            b.set_value(b.get_value() + db)

            print('{0:>6} {1:.4f}'.format(t, np.mean(values[-100:])))

    except KeyboardInterrupt:
        pass

    return A.get_value() * np.eye(data.shape[0]), b.get_value().T


def js(X, Z, G, log_p, log_q, a=0.0):
    return (tt.mean(tt.log(tt.nnet.sigmoid(log_p(X) - log_q(G(Z)))))
            + tt.mean(tt.log(tt.nnet.sigmoid(log_q(G(Z)) - log_p(X))))
            + tt.mean(0.0 * a * G(Z)))  # Use dummy vars


def kl(X, Z, G, log_p, log_q, a=0.0):
    return (tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
            + tt.mean(0.0 * a * G(Z)))  # Use dummy vars


def rkl(X, Z, G, log_p, log_q, a=0.0):
    return (tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(X)))
            + tt.mean(0.0 * a * G(Z)))  # Use dummy vars


def gjs(X, Z, G, log_p, log_q, a=0.5):
    return ((1 - a) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(G(Z))))
            + a ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(X))))


def dgjs(X, Z, G, log_p, log_q, a=0.5):
    return ((1 - a) ** 2 * tt.mean((tt.exp(log_p(X)) ** a)
                                   * (tt.exp(log_q(G(Z))) ** (1 - a))
                                   * (log_q(G(Z)) - log_p(X)))
            + a ** 2 * tt.mean((tt.exp(log_p(X))) ** a
                               * (tt.exp(log_q(G(Z))) ** (1 - a))
                               * (log_p(X) - log_q(G(Z)))))


def fit(data, log_p, div='kl', max_epochs=20, alpha=0.5):
    D = data.shape[0]
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # Initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None],
                  broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None],
                  broadcastable=[False, True])
    if div in ['tgjs', 'tdgjs']:
        alpha = th.shared(alpha)

    def log_q(X):
        return (-0.5 * tt.sum(tt.square((X - b) / a), 0)
                - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi))

    def G(Z):
        return a * Z + b

    if div in ['tgjs', 'tdgjs']:
        div = eval(div)(X, Z, G, log_p, log_q, a=alpha)
        f_div = th.function(
            [Z, X],
            [div, th.grad(div, a), th.grad(div, b), th.grad(div, alpha)])
    else:
        div = eval(div)(X, Z, G, log_p, log_q, a=alpha)
        f_div = th.function(
            [Z, X],
            [div, th.grad(div, a), th.grad(div, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = 0.5
    da = 0.0
    db = 0.0

    print('{0:>4} {1:.4f}'.format(0, f_div(nr.randn(*data.shape), data)[0]))
    for epoch in range(max_epochs):
        values = []
        for t in range(0, data.shape[1], B):  # SGD with momentum
            Z = nr.randn(D, B)
            Y = data[:, t:t + B]
            v, ga, gb = f_div(Z, Y)
            da = mm * da - lr * ga
            db = mm * db - lr * gb

            a.set_value(a.get_value() + da)
            b.set_value(b.get_value() + db)
            values.append(v)
        # lr /= 2.
        print(f'{epoch+1:>4} {np.mean(values):.4f}')

    return a.get_value() * np.eye(D), b.get_value()


def fit_js(data, log_p, max_epochs=20):
    """
    Fit isotropic Gaussian by minimizing Jensen-Shannon divergence.
    """

    # data dimensionality
    D = data.shape[0]

    # data and hidden states
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])

    # model density
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    def G(Z): return a * Z + b

    # Jensen-Shannon divergence
    JSD = (tt.mean(tt.log(tt.nnet.sigmoid(log_p(X) - log_q(G(Z)))))
           + tt.mean(tt.log(tt.nnet.sigmoid(log_q(G(Z)) - log_p(X)))))

    JSD = (JSD + np.log(4.)) / 2.
    # JSD1 = tt.mean(tt.log(tt.nnet.sigmoid(log_p(X) - log_q(X))))
    # JSD2 = tt.mean(tt.log(tt.nnet.sigmoid(log_q(G(Z)) - log_p(G(Z)))))
    # JSD = l * JSD1 + (1 - l) * JSD2

    # function computing JSD and its gradient
    f_jsd = th.function([Z, X], [JSD, th.grad(JSD, a), th.grad(JSD, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5

    da = 0.
    db = 0.

    try:
        # display initial JSD
        print('{0:>4} {1:.4f}'.format(0, float(f_jsd(nr.randn(*data.shape), data)[0])))

        for epoch in range(max_epochs):
            values = []

            # stochastic gradient descent
            for t in range(0, data.shape[1], B):
                Z = nr.randn(D, B)
                Y = data[:, t:t + B]

                v, ga, gb = f_jsd(Z, Y)
                da = mm * da - lr * ga
                db = mm * db - lr * gb

                values.append(v)

                a.set_value(a.get_value() + da)
                b.set_value(b.get_value() + db)

            # reduce learning rate
            lr /= 2.

            # display estimated JSD
            print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    except KeyboardInterrupt:
        pass

    return a.get_value() * np.eye(D), b.get_value()


def fit_gjs_train_a(data, log_p, max_epochs=50):
    """Fit isotropic Gaussian by minimizing GJS divergence."""
    D = data.shape[0]

    # Data and hidden states
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # Initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])
    alpha = th.shared(0.5)

    # Density and divergence
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)
    def G(Z): return a * Z + b
    gJSD = ((1 - alpha) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(G(Z))))
            + alpha ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(X))))

    # Function computing G-JSD and its gradient
    f_gjsd = th.function(
        [Z, X],
        [gJSD, th.grad(gJSD, a), th.grad(gJSD, b), th.grad(gJSD, alpha)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .002
    da = 0.
    db = 0.
    dalpha = 0.

    print('{0:>4} {1:.4f}'.format(
        0, float(f_gjsd(nr.randn(*data.shape), data)[0])))
    print("Starting training! \n\n")
    for epoch in range(max_epochs):
        values = []
        print(f"Alpha: {alpha.get_value()}")
        for t in range(0, data.shape[1], B):

            Z = nr.randn(D, B)
            Y = data[:, t:t + B]
            v, ga, gb, galpha = f_gjsd(Z, Y)
            da = mm * da - lr * ga
            db = mm * db - lr * gb
            dalpha = mm * dalpha - lr * galpha

            values.append(v)

            a.set_value(a.get_value() + da)
            b.set_value(b.get_value() + db)
            alpha.set_value(alpha.get_value() + dalpha)
            if alpha.get_value() > 1.0:
                alpha.set_value(1.0)
            elif alpha.get_value() < 0.0:
                alpha.set_value(0.0)

        # lr /= 2.0
        print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    return a.get_value() * np.eye(D), b.get_value(), alpha.get_value()


def fit_gjs(data, log_p, max_epochs=20):
    """Fit isotropic Gaussian by minimizing geometric Jensen-Shannon divergence."""
    D = data.shape[0]
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])

    alpha = 0.4
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)
    def G(Z): return a * Z + b

    # geometric Jensen-Shannon divergence
    # JSD = tt.mean(log_p(X) - log_q(X)) \
    #     + tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
    gJSD = ((1 - alpha) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(G(Z))))
            + alpha ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(X))))
    f_gjsd = th.function([Z, X], [gJSD, th.grad(gJSD, a), th.grad(gJSD, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5
    da = 0.
    db = 0.

    print('{0:>4} {1:.4f}'.format(0, float(f_gjsd(nr.randn(*data.shape), data)[0])))
    for epoch in range(max_epochs):
        values = []
        print(f"Alpha: {alpha}")
        # stochastic gradient descent
        for t in range(0, data.shape[1], B):
            Z = nr.randn(D, B)
            Y = data[:, t:t + B]
            v, ga, gb = f_gjsd(Z, Y)
            da = mm * da - lr * ga
            db = mm * db - lr * gb

            values.append(v)

            a.set_value(a.get_value() + da)
            b.set_value(b.get_value() + db)
        # lr /= 2.
        print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    return a.get_value() * np.eye(D), b.get_value()


def fit_dgjs(data, log_p, max_epochs=20):
    """
    Fit isotropic Gaussian by minimizing geometric Jensen-Shannon divergence.
    """

    # data dimensionality
    D = data.shape[0]

    # data and hidden states
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])
    # alpha = th.shared(0.5)

    # model density
    def q(X): return normal(X, b, a)
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    def G(Z): return a * Z + b

    # geometric Jensen-Shannon divergence
    # JSD = tt.mean(log_p(X) - log_q(X)) \
    #     + tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
    # gJSD = (1 - 0.5) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(X))) \
    #     + 0.5 ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
    # gJSD = (1 - alpha) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(X))) + alpha ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
    
    alpha = 1.0

    # gJSD = (1 - 0.5) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(G(Z))))                                                   + 0.5 ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(X)))
    gJSD = (1 - alpha) ** 2 * tt.mean((tt.exp(log_p(X)) ** (alpha)) * (tt.exp(log_q(G(Z))) ** (1 - alpha)) * (log_q(G(Z)) - log_p(X)))   + alpha ** 2 * tt.mean((tt.exp(log_p(X))) ** (alpha) * (tt.exp(log_q(G(Z))) ** (1 - alpha)) * (log_p(X) - log_q(G(Z))))

    # function computing G-JSD and its gradient
    # f_gjsd = th.function([Z, X], [gJSD, th.grad(gJSD, a), th.grad(gJSD, b), th.grad(gJSD, alpha)])
    f_gjsd = th.function([Z, X], [gJSD, th.grad(gJSD, a), th.grad(gJSD, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5

    da = 0.
    db = 0.
    dalpha = 0.
    
    try:
        # display initial JSD
        print('{0:>4} {1:.4f}'.format(0, float(f_gjsd(nr.randn(*data.shape), data)[0])))

        for epoch in range(max_epochs):
            values = []
            print(f"Alpha: {alpha}")
            # stochastic gradient descent
            for t in range(0, data.shape[1], B):
                
                Z = nr.randn(D, B)
                Y = data[:, t:t + B]

                # v, ga, gb, galpha = f_gjsd(Z, Y)
                # da = mm * da - lr * ga
                # db = mm * db - lr * gb
                # dalpha = mm * dalpha - lr * galpha
                v, ga, gb = f_gjsd(Z, Y)
                da = mm * da - lr * ga
                db = mm * db - lr * gb

                values.append(v)

                a.set_value(a.get_value() + da)
                b.set_value(b.get_value() + db)
                

            # reduce learning rate
            lr /= 2.

            # display estimated JSD
            print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    except KeyboardInterrupt:
        pass
    return a.get_value() * np.eye(D), b.get_value()


def fit_kl(data, log_p, max_epochs=20):
    """
    Fit isotropic Gaussian by minimizing reverse Kullback-Leibler divergence.
    """

    # data dimensionality
    D = data.shape[0]

    # data and hidden states
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])

    # model density
    def q(X): return normal(X, b, a)
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    def G(Z): return a * Z + b

    # geometric Jensen-Shannon divergence
    KL = tt.mean(0.0 * X) + tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))

    # function computing G-JSD and its gradient
    f_kl = th.function([Z, X], [KL, th.grad(KL, a), th.grad(KL, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5

    da = 0.
    db = 0.

    try:
        # display initial JSD
        print('{0:>4} {1:.4f}'.format(0, float(f_kl(nr.randn(*data.shape), data)[0])))

        for epoch in range(max_epochs):
            # print(f'\nEpoch: {epoch}')
            values = []

            # stochastic gradient descent
            for t in range(0, data.shape[1], B):
                Z = nr.randn(D, B)
                Y = data[:, t:t + B]

                v, ga, gb = f_kl(Z, Y)
                da = mm * da - lr * ga
                db = mm * db - lr * gb

                values.append(v)

                a.set_value(a.get_value() + da)
                b.set_value(b.get_value() + db)
                # reduce learning rate
                lr /= 2.

            # display estimated JSD
            print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    except KeyboardInterrupt:
        pass

    return a.get_value() * np.eye(D), b.get_value()


def fit_rkl(data, log_p, max_epochs=20):
    """
    Fit isotropic Gaussian by minimizing reverse Kullback-Leibler divergence.
    """

    # data dimensionality
    D = data.shape[0]

    # data and hidden states
    X = tt.dmatrix('X')
    Z = tt.dmatrix('Z')

    nr.seed(int(time() * 1000.) % 4294967295)
    idx = nr.permutation(data.shape[1])[:100]

    # initialize parameters
    b = th.shared(np.mean(data[:, idx], 1)[:, None], broadcastable=(False, True))
    a = th.shared(np.std(data[:, idx] - b.get_value(), 1)[:, None], broadcastable=[False, True])

    # model density
    def q(X): return normal(X, b, a)
    def log_q(X): return -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    def G(Z): return a * Z + b

    # geometric Jensen-Shannon divergence
    RKL = tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(X))) + tt.mean(0.0 * Z)

    # function computing G-JSD and its gradient
    f_rkl = th.function([Z, X], [RKL, th.grad(RKL, a), th.grad(RKL, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5

    da = 0.
    db = 0.

    try:
        # display initial JSD
        print('{0:>4} {1:.4f}'.format(0, float(f_rkl(nr.randn(*data.shape), data)[0])))

        for epoch in range(max_epochs):
            values = []

            # stochastic gradient descent
            for t in range(0, data.shape[1], B):
                Z = nr.randn(D, B)
                Y = data[:, t:t + B]

                v, ga, gb = f_rkl(Z, Y)
                da = mm * da - lr * ga
                db = mm * db - lr * gb

                values.append(v)

                a.set_value(a.get_value() + da)
                b.set_value(b.get_value() + db)

            # reduce learning rate
            lr /= 2.

            # display estimated JSD
            print('{0:>4} {1:.4f}'.format(epoch + 1, np.mean(values)))

    except KeyboardInterrupt:
        pass

    return a.get_value() * np.eye(D), b.get_value()


def main(argv):
    parser = ArgumentParser(
        argv[0],
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--divergence', '-m',
        choices=['mmd', 'kl', 'rkl', 'js', 'gjs', 'dgjs', '', 'tgjs'],
        default='kl',
        help='Which metric to use.')
    parser.add_argument(
        '--num_data', '-N', type=int, default=100000,
        help='Number of training points.')
    parser.add_argument(
        '-d', type=int, default=2,
        help='Dimension of optimisation problem.')
    parser.add_argument(
        '-k', type=int, default=10,
        help='Number of mixture components.')
    parser.add_argument(
        '-a', type=float, default=0.5,
        help='Initial alpha for skew divergences')
    parser.add_argument(
        '--seed', '-s', type=int, default=22,
        help='Random seed used to generate data.')
    parser.add_argument(
        '--output', '-o', type=str, default='results/',
        help='Where to store results.')
    args = parser.parse_args(argv[1:])

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print('Generating data...')

    p, log_p, data = mogaussian(
        D=args.d, K=args.k, N=args.num_data, seed=args.seed)
    print(f'Optimizing {args.divergence} divergence...')

    A, b = fit(data, log_p, div=args.divergence, alpha=args.a)

    # if 'KL' in args.metrics:
    #     A, b = fit_kl(data, log_p)

    # if 'RKL' in args.metrics:
    #     A, b = fit_rkl(data, log_p)

    # if 'JS' in args.metrics:
    #     A, b = fit_js(data, log_p)

    # if 'GJS' in args.metrics:
    #     A, b = fit_gjs(data, log_p)

    # if 'tGJS' in args.metrics:
    #     A, b, alpha = fit_gjs_train_a(data, log_p)
    #     # np.savetxt(
    #     #     os.path.join(
    #     #         args.output,
    #     #         f'tGJS_d={args.d}_k={args.k}_{args.seed}.png'),
    #     #     np.array(alpha)[None, None])

    # if 'dGJS' in args.metrics:
    #     A, b = fit_dgjs(data, log_p)

    # if 'MMD' in args.metrics:
    #     A, b = fit_mmd(data)

    if args.d == 2:
        plot(log_p, data)
        plt.savefig(os.path.join(args.output, f'{args.seed}_data.png'))
        for metric in args.metrics:
            plot([A, b], data)
            plt.savefig(
                os.path.join(
                    args.output,
                    f'{metric}_d={args.d}_k={args.k}_{args.seed}.png'))


if __name__ == '__main__':
    main(sys.argv)
