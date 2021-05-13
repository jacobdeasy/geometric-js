import abc
import json
import logging
import numpy as np
import os
import seaborn as sns
import sys
import torch
import torch.optim as optim

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from torch import Tensor
from tqdm import tqdm
from typing import Any, List, Optional, Tuple

sns.set()
sns.set_style('whitegrid')
sns.set_style('ticks')

TRAIN_LOSSES_LOGFILE = "train_losses.log"


class BaseDivergence(abc.ABC):

    def __init__(self,
                 dist_params: List,
                 sample_size: Optional[int] = 200,
                 initial_mean: Optional[float] = None,
                 dimensions: Optional[int] = 2,
                 **kwargs: Optional[Any]
                 ) -> None:
        super().__init__(**kwargs)
        if initial_mean is not None:
            self.mean = torch.tensor(np.array(initial_mean)[:, None]).float()
            self.mean = torch.nn.Parameter(self.mean)
        else:
            self.mean = torch.nn.Parameter(torch.ones((dimensions, 1)).float())
        self.covariance = torch.nn.Parameter(torch.eye(dimensions))
        self.dist_params = dist_params
        self.dimensions = dimensions
        self.sample_size = sample_size

    def p(self, X: Tensor) -> Tensor:
        """Additive gaussian mixture model probabilities."""
        total_probability = torch.zeros(self.sample_size, 1)
        for params in self.dist_params:
            mean, covariance, weight = params
            mean = torch.tensor(np.array([m for m in mean])[:, None]).float()
            covariance = torch.tensor(covariance).float()
            probabilities = self.normal(X, mean, covariance)
            total_probability += weight * probabilities

        return total_probability

    def log_p(self, X: Tensor) -> Tensor:
        return self.p(X).log()

    def q(self, X: Tensor) -> Tensor:
        """Gaussian distribution."""
        return self.normal(X, self.mean, self.covariance)

    def log_q(self, X: Tensor) -> Tensor:
        return self.log_normal(X, self.mean, self.covariance)

    def normal(self, X: Tensor, m: Tensor, C: Tensor) -> Tensor:
        Z = X - m

        return torch.exp(
            - torch.sum(Z * torch.matmul(torch.inverse(C), Z), 1) / 2.
            - torch.log(torch.det(C)) / 2.
            - len(m) / 2. * torch.log(2. * torch.tensor(np.pi)))

    def log_normal(self, X: Tensor, m: Tensor, C: Tensor) -> Tensor:
        Z = X - m

        return (- torch.sum(Z * torch.matmul(torch.inverse(C), Z), 1) / 2.
                - torch.log(torch.det(C)) / 2.
                - len(m) / 2. * torch.log(2. * torch.tensor(np.pi)))


class fwdKL(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='./',
                 **kwargs):
        super(fwdKL, self).__init__(**kwargs)

        # Logger for storing the parameter values during training:
        if not os.path.exists(f"{train_data_file_path}/fwdKL"):
            print("Creating folder!!")
            os.makedirs(f"{train_data_file_path}/fwdKL")
        elif os.path.isfile(
                f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}"):
            os.remove(f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}")
        file_path = f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}"

        print(f"Logging file path: {file_path}")
        self.logger = logging.getLogger("losses_logger_fwdKL")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def forward(self, X):
        return torch.sum(self.log_q(X).exp() * (self.log_q(X) - self.log_p(X)))

    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")

        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")

        for i, var in enumerate(
                self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


class revKL(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='./',
                 **kwargs):
        super(revKL, self).__init__(**kwargs)

        # Logger for storing the parameter values during training:
        if not os.path.exists(f"{train_data_file_path}/revKL"):
            print("Creating folder!!")
            os.makedirs(f"{train_data_file_path}/revKL")
        elif os.path.isfile(
                f"{train_data_file_path}/revKL/{TRAIN_LOSSES_LOGFILE}"):
            os.remove(f"{train_data_file_path}/revKL/{TRAIN_LOSSES_LOGFILE}")
        file_path = f"{train_data_file_path}/revKL/{TRAIN_LOSSES_LOGFILE}"

        print(f"Logging file path: {file_path}")
        self.logger = logging.getLogger("losses_logger_revKL")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def forward(self, X):
        return torch.sum(self.p(X) * (self.log_p(X) - self.log_q(X)))

    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")

        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")

        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


class JS(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='./',
                 **kwargs):
        super(JS, self).__init__(**kwargs)

        # Logger initialisation:

        if not os.path.exists(f"{train_data_file_path}/JS"):
            print("Creating folder!!")
            os.makedirs(f"{train_data_file_path}/JS")
        elif os.path.isfile(f"{train_data_file_path}/JS/{TRAIN_LOSSES_LOGFILE}"):
            os.remove(f"{train_data_file_path}/JS/{TRAIN_LOSSES_LOGFILE}")
        file_path = f"{train_data_file_path}/JS/{TRAIN_LOSSES_LOGFILE}"

        print(f"Logging file path: {file_path}")
        self.logger = logging.getLogger("losses_logger_JS")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def forward(self, sample_data):
        loss = 0.5 * torch.sum(self.q(sample_data) * (self.log_q(sample_data) \
                - (0.5 * self.p(sample_data) + 0.5 * self.q(sample_data)).log())) \
                + 0.5 * torch.sum(self.p(sample_data) * (self.log_p(sample_data)
                - (0.5 * self.p(sample_data) + 0.5 * self.q(sample_data)).log()))
        return loss

    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")

        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")

        for i, var in enumerate(
                self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


class GJS(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 alpha=0.5,
                 dual=False,
                 train_data_file_path='./',
                 **kwargs):
        super(GJS, self).__init__(**kwargs)
        self.alpha = alpha
        self.dual = dual

        # Logger for storing the parameter values during training:
        log_folder = f"{train_data_file_path}/{'GJS' if dual == False else 'dGJS'}-A_0={alpha}"
        log_file = f"{train_data_file_path}/{'GJS' if dual == False else 'dGJS'}-A_0={alpha}/{TRAIN_LOSSES_LOGFILE}"
        print(f"Logging file path: {log_file}")

        if not os.path.exists(log_folder):
            print("Creating folder!!")
            os.makedirs(log_folder)
        elif os.path.isfile(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger("losses_logger_GJS")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def forward(self, sample_data):
        qx = self.q(sample_data)
        px = self.p(sample_data)
        if self.dual:
            KL_1 = torch.sum(qx.pow(self.alpha) * px.pow(1 - self.alpha) * (px.log() - qx.log()))
            KL_2 = torch.sum(qx.pow(self.alpha) * px.pow(1 - self.alpha) * (qx.log() - px.log()))
            loss = ((1 - self.alpha) ** 2) * KL_1 + (self.alpha ** 2) * KL_2
        else:
            KL_1 = torch.sum(qx * (qx.log() - px.log()))
            KL_2 = torch.sum(px * (px.log() - qx.log()))
            loss = ((1 - self.alpha) ** 2) * KL_1 + (self.alpha ** 2) * KL_2
        return loss

    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},alpha,{self.alpha}")
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")

        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")

        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


class GJSTrainAlpha(BaseDivergence, torch.nn.Module):

    def __init__(self,
                 alpha: Optional[float] = 0.5,
                 dual: Optional[bool] = False,
                 train_data_file_path: Optional[str] = './',
                 **kwargs: Optional[Any]
                 ) -> None:
        super(GJSTrainAlpha, self).__init__(**kwargs)
        self.a = torch.nn.Parameter(torch.tensor(alpha))
        self.dual = dual

        folder_name = f"{'tGJS' if dual == False else 'tdGJS'}-A_0={alpha}"
        log_folder = os.path.join(train_data_file_path, folder_name)
        log_file = os.path.join(log_folder, TRAIN_LOSSES_LOGFILE)
        print(f"Logging file path: {log_file}")

        if not os.path.exists(log_folder):
            print("Creating folder!!")
            os.makedirs(log_folder)
        elif os.path.isfile(log_file):
            os.remove(log_file)

        self.logger = logging.getLogger(f"losses_logger_{folder_name}")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def forward(self, sample_data: torch.Tensor) -> torch.Tensor:
        lqx = self.log_q(sample_data)
        lpx = self.log_p(sample_data)
        if self.dual:
            kl_1 = torch.mean(
                lqx.exp() ** self.a * lpx.exp() ** (1 - self.a) * (lpx - lqx))
            kl_2 = torch.mean(
                lqx.exp() ** self.a * lpx.exp() ** (1 - self.a) * (lqx - lpx))
            loss = (1 - self.a) ** 2 * kl_1 + self.a ** 2 * kl_2
        else:
            kl_1 = torch.mean(lqx.exp() * (lpx - lqx))
            kl_2 = torch.mean(lpx.exp() * (lpx - lqx))
            loss = (1 - self.a) ** 2 * kl_1 + self.a ** 2 * kl_2

        return loss

    def log(self, epoch, av_divergence):
        self.logger.debug(f"{epoch},alpha,{self.a.item()}")
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")

        for i, m in enumerate(self.mean.detach().numpy()):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")

        for i, var in enumerate(
                self.covariance.diag().detach().numpy()):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


def get_sample_data(save_loc: str,
                    seed: Optional[int] = 1234,
                    dimensions: Optional[int] = 2,
                    num_gaussians: Optional[int] = 5,
                    num_samples: Optional[int] = 100000,
                    save: Optional[bool] = False
                    ) -> Tuple[List, np.ndarray]:
    """
    Function which produces a ND add-mixture of Gaussian distribution, with
    num_gaussians number of gaussian components, each of which has parameters
    radomly generated. The parameters of each component are the mean,
    covariance and weight.

    Returns
    -------
    dist_parms : list
        Contains the parameters of each component in the mixture distribution.
        Each entry in the list is a vector containing the average coordinates,
        the covariance matrix and weight of the Gaussan component.

    data_sampes : numpy array
        Array of shape (num_samples, 2) of data coordinates randomly sampled
        from the mixture distribution.
    """
    np.random.seed(seed)

    p = np.random.dirichlet([0.5] * num_gaussians)
    v = 1 / np.square(np.random.rand(num_gaussians) + 1)
    C = [np.eye(dimensions) * v_i for v_i in v]

    dist_params = [[[np.random.rand() * 6 - 3 for i in range(dimensions)],
                    covariance, weight] for covariance, weight in zip(C, p)]
    data_samples = np.concatenate(
        [multivariate_normal.rvs(mean=m,
                                 cov=covariance,
                                 size=max(int(num_samples * weight), 2))
            for m, covariance, weight in dist_params])

    if save and dimensions == 2:
        x, y = np.mgrid[-6:6:.01, -6:6:.01]
        pos = np.dstack((x, y))
        pdf = np.array([weight * multivariate_normal(mean, covariance).pdf(pos)
                        for mean, covariance, weight in dist_params])

        fig = plt.figure(figsize=(10, 10))
        ax2 = fig.add_subplot(111)
        ax2.contour(x, y, pdf.sum(axis=0)/pdf.sum(axis=0).sum())
        plt.scatter(data_samples[:, 0], data_samples[:, 1], s=0.01, c='k')
        if not os.path.exists(f"{save_loc}"):
            os.makedirs(f"{save_loc}")
        plt.axis('equal')
        plt.axis([np.min(data_samples[:, 0]),
                  np.max(data_samples[:, 0]),
                  np.min(data_samples[:, 1]),
                  np.max(data_samples[:, 1])])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().tight_layout()
        plt.savefig(f"{save_loc}/mixture-distribution.png", dpi=200)

    return dist_params, data_samples


def train_model(model,
                optimizer,
                data_samples,
                log_loc,
                learn_a: Optional[bool] = False,
                dimensions: Optional[int] = 2,
                epochs: Optional[int] = 20,
                sample_size: Optional[int] = 200,
                name: Optional[str] = None,
                save_loc: Optional[str] = None,
                frequency: Optional[int] = 2
                ) -> None:
    """
    Trains a passed model to learn the parameters of a
    multivariate Gaussian distribution to approximate a mixture multivariate
    Gaussian.
    """
    for e in range(epochs):
        divergence = 0
        np.random.shuffle(data_samples)
        print(f"Epoch {e + 1}")
        for i in tqdm(range(int(len(data_samples) / sample_size))):

            sample = torch.tensor(data_samples[i:i+sample_size]).view(
                sample_size, dimensions, 1).float()

            loss = model(sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            divergence += loss.detach()

            with torch.no_grad():
                if learn_a:
                    model.a.clamp_(0, 1)

                # Enforce the covariance matrix to diagonal:
                for i in range(dimensions):
                    for j in range(dimensions):
                        if i == j:
                            continue
                        else:
                            model.covariance[i][j] = 0.0
        model.log(e, divergence / len(data_samples) / sample_size)

    if save_loc is not None and name is not None and dimensions == 2:
        x, y = np.mgrid[-6:6:.01, -6:6:.01]
        pos = np.dstack((x, y))
        pdf = multivariate_normal(
            model.mean.detach().numpy().reshape(-1),
            model.covariance.detach().numpy()).pdf(pos)
        fig = plt.figure(figsize=(10, 10))
        ax2 = fig.add_subplot(111)
        ax2.contour(x, y, pdf)
        plt.scatter(data_samples[:, 0], data_samples[:, 1], s=0.01, c='k')
        plt.axis('equal')
        plt.axis([
            np.min(data_samples[:, 0]),
            np.max(data_samples[:, 0]),
            np.min(data_samples[:, 1]),
            np.max(data_samples[:, 1])])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().tight_layout()
        plt.savefig(f"{save_loc}/fitted-dist-{name}.png", dpi=200)

    # Save learnt parameters:
    learnt_dist = {}
    if learn_a:
        learnt_dist["alpha"] = str(model.a.item())
    elif hasattr(model, 'alpha'):
        learnt_dist["alpha"] = str(model.alpha)
    for i, m in enumerate(model.mean.detach().numpy()):
        learnt_dist[f"mean_{i + 1}"] = str(m)
    for i, var in enumerate(model.covariance.diag().detach().numpy()):
        learnt_dist[f"var_{i+1}"] = str(var)
    with open(f'{log_loc}/{name}/final-parameters.txt', 'w') as f:
        json.dump(learnt_dist, f)


def main(argv):
    parser = ArgumentParser(argv[0],
                            description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--metric', '-m',
        choices=['fwdKL', 'revKL', 'JS', 'GJS', 'dGJS', 'tGJS', 'tdGJS'],
        help='Which metric to optimise.')
    parser.add_argument(
        '--num-data', '-N', type=int, default=100000,
        help='Number of points in the sampled data from the mixture.')
    parser.add_argument(
        '--sample-size', '-S', type=int, default=200,
        help='Number of points in the each training batch.')
    parser.add_argument(
        '--num-mixture', type=int, default=5,
        help='Number of Gaussian components in the mixture distribution.')
    parser.add_argument(
        '--dimensions', type=int, default=2,
        help='Dimensions of the Gaussian distributions being modelled.')
    parser.add_argument(
        '--seed', '-s', type=int, default=22,
        help='Random seed used to generate data.')
    parser.add_argument(
        '--A0', type=float, default=0.5,
        help='Initial value of alpha if training GJS or dGJS.')
    parser.add_argument(
        '--lr', '--learning-rate', type=float, default=1e-4,
        help='Learning rate.')
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs.')
    parser.add_argument(
        '--plot-output', type=str, default=os.path.join(os.pardir, 'figs'),
        help='Where to store plots produced.')
    parser.add_argument(
        '--train-log-output', '-o', type=str,
        default=os.path.join(os.pardir, 'logs'),
        help='Where to store log of data produced during training.')
    parser.add_argument(
        '--save-mixture-plot', type=bool, default=True,
        help='Where to store results.')
    args = parser.parse_args(argv[1:])

    plot_loc = f"{args.plot_output}/{args.seed}-{args.dimensions}" + \
               f"-{args.num_mixture}-{args.lr}-{args.epochs}"
    log_loc = f"{args.train_log_output}/{args.seed}-{args.dimensions}" + \
              f"-{args.num_mixture}-{args.lr}-{args.epochs}"
    name = f"{args.metric}"

    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)
    if not os.path.exists(log_loc):
        os.makedirs(log_loc)

    print('Generating data...')
    dist_params, data_samples = get_sample_data(dimensions=args.dimensions,
                                                save_loc=plot_loc,
                                                save=True,
                                                seed=args.seed,
                                                num_gaussians=args.num_mixture,
                                                num_samples=args.num_data)

    if args.metric == 'tGJS' or args.metric == 'tdGJS':
        print('Optimizing trainable-alpha GJS divergence...')
        model = GJSTrainAlpha(dist_params=dist_params,
                              dimensions=args.dimensions,
                              sample_size=args.sample_size,
                              dual=args.metric == 'tdGJS',
                              train_data_file_path=log_loc,
                              alpha=args.A0)
        name = f"{args.metric}-A_0={args.A0}"

    if args.metric == 'dGJS' or args.metric == 'GJS':
        print('Optimizing constant-alpha GJS divergence...')
        model = GJS(dist_params=dist_params,
                    dimensions=args.dimensions,
                    sample_size=args.sample_size,
                    dual=args.metric == 'dGJS',
                    train_data_file_path=log_loc,
                    alpha=args.A0)

    if args.metric == 'JS':
        print('Optimizing Jensen-Shannon divergence...')
        model = JS(dist_params=dist_params,
                   dimensions=args.dimensions,
                   sample_size=args.sample_size,
                   train_data_file_path=log_loc)

    if args.metric == 'revKL':
        print('Optimizing Reverse Kullback-Leibler divergence...')
        model = revKL(dist_params=dist_params,
                      dimensions=args.dimensions,
                      sample_size=args.sample_size,
                      train_data_file_path=log_loc)

    if args.metric == 'fwdKL':
        print('Optimizing Kullback-Leibler  divergence...')
        model = fwdKL(dist_params=dist_params,
                      dimensions=args.dimensions,
                      sample_size=args.sample_size,
                      train_data_file_path=log_loc)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model(model,
                optimizer,
                data_samples,
                log_loc=log_loc,
                learn_a=args.metric[0] == 't',
                dimensions=args.dimensions,
                epochs=args.epochs,
                sample_size=args.sample_size,
                name=name,
                save_loc=plot_loc)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
