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
import pdb
import theano.sandbox.linalg as tl
import theano.tensor as tt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.optimize import minimize
from time import time

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
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200))
    zz = np.exp(log_q(np.asarray([xx.ravel(), yy.ravel()])).reshape(xx.shape))

    hh, x, y = np.histogram2d(data[0], data[1], 80, range=[(xmin, xmax), (ymin, ymax)])

    sns.set_style('whitegrid')
    sns.set_style('ticks')
    plt.figure(figsize=(10, 10), dpi=300)
    # plt.imshow(hh.T[::-1], extent=[x[0], x[-1], y[0], y[-1]],
    # 	interpolation='nearest', cmap='YlGnBu_r')
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
    log_q = lambda X: -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    G = lambda Z: a * Z + b

    # Jensen-Shannon divergence
    JSD = tt.mean(tt.log(tt.nnet.sigmoid(log_p(X) - log_q(X)))) \
        + tt.mean(tt.log(tt.nnet.sigmoid(log_q(G(Z)) - log_p(G(Z)))))
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


def fit_gjs(data, log_p, max_epochs=20):
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
    
    # model density
    q = lambda X: normal(X, b, a)
    log_q = lambda X: -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    G = lambda Z: a * Z + b

    # geometric Jensen-Shannon divergence
    # JSD = tt.mean(log_p(X) - log_q(X)) \
    #     + tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))
    gJSD = (1 - 0.5) ** 2 * tt.mean(tt.exp(log_p(X)) * (log_p(X) - log_q(X))) \
        + 0.5 ** 2 * tt.mean(tt.exp(log_q(G(Z))) * (log_q(G(Z)) - log_p(G(Z))))

    # function computing G-JSD and its gradient
    f_gjsd = th.function([Z, X], [gJSD, th.grad(gJSD, a), th.grad(gJSD, b)])

    # SGD hyperparameters
    B = 200
    mm = 0.8
    lr = .5

    da = 0.
    db = 0.

    try:
        # display initial JSD
        print('{0:>4} {1:.4f}'.format(0, float(f_gjsd(nr.randn(*data.shape), data)[0])))

        for epoch in range(max_epochs):
            values = []

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
    q = lambda X: normal(X, b, a)
    log_q = lambda X: -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    G = lambda Z: a * Z + b

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
    q = lambda X: normal(X, b, a)
    log_q = lambda X: -0.5 * tt.sum(tt.square((X - b) / a), 0) - D * tt.log(tt.abs_(a)) - D / 2. * np.log(np.pi)

    G = lambda Z: a * Z + b

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
    parser = ArgumentParser(argv[0],
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--metrics', '-m', choices=['MMD', 'KL', 'RKL', 'JS', 'GJS', ''], nargs='+', default=['KL', 'RKL', 'JS', 'GJS'],
        help='Which metrics to include in comparison.')
    parser.add_argument('--num_data', '-N', type=int, default=100000,
        help='Number of training points.')
    parser.add_argument('--seed', '-s', type=int, default=22,
        help='Random seed used to generate data.')
    parser.add_argument('--output', '-o', type=str, default='results/',
        help='Where to store results.')

    args = parser.parse_args(argv[1:])

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print('Generating data...')

    nonlog_p, log_p, data = mogaussian(D=2, N=args.num_data, seed=args.seed)

    plot(log_p, data)
    plt.savefig(os.path.join(args.output, '{0}_data.png'.format(args.seed)))

    # if 'KL' in args.metrics:
    #     print('Optimizing Kullback-Leibler divergence...')

    #     b = np.mean(data, 1)[:, None]
    #     A = np.eye(2) * np.std(data - b, 1)[:, None]

    #     plot([A, b], data)
    #     plt.savefig(os.path.join(args.output, '{0}_KL.png'.format(args.seed)))

    if 'KL' in args.metrics:
        print('Optimizing Kullback-Leibler divergence...')

        A, b = fit_kl(data, log_p)

        plot([A, b], data)
        plt.savefig(os.path.join(args.output, '{0}_KL.png'.format(args.seed)))

    if 'RKL' in args.metrics:
        print('Optimizing *reverse* Kullback-Leibler divergence...')

        A, b = fit_rkl(data, log_p)

        plot([A, b], data)
        plt.savefig(os.path.join(args.output, '{0}_RKL.png'.format(args.seed)))

    if 'JS' in args.metrics:
        print('Optimizing Jensen-Shannon divergence...')

        A, b = fit_js(data, log_p)

        plot([A, b], data)
        plt.savefig(os.path.join(args.output, '{0}_JS.png'.format(args.seed)))

    if 'GJS' in args.metrics:
        print('Optimizing *geometric* Jensen-Shannon divergence...')

        A, b = fit_gjs(data, log_p)

        plot([A, b], data)
        plt.savefig(os.path.join(args.output, '{0}_GJS.png'.format(args.seed)))

    if 'MMD' in args.metrics:
        print('Optimizing MMD...')

        A, b = fit_mmd(data)

        plot([A, b], data)
        plt.savefig(os.path.join(args.output, '{0}_MMD.png'.format(args.seed)))

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
