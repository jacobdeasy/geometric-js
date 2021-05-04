import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.optim as optim
import json
import sys
import os
import logging
import abc
from tqdm import tqdm
from timeit import default_timer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('whitegrid')
sns.set_style('ticks')

TRAIN_LOSSES_LOGFILE = "train_losses.log"

# Divergence Loss Classes:

class BaseDivergence(abc.ABC):
    """
    Base class containing function definitions for probability density
    calculators, for both the learnt distribution and the distribution
    being modelled.
    """

    def __init__(self,
                 dist_params,
                 sample_size=200,
                 initial_mean=None,
                 dimensions=2,
                 **kwargs):
        super().__init__(**kwargs)
        """
        Parameters
        ----------
        dist_params : list 
            Each entry should be a tuple or list of parameters specifying
            the parameters of each Gaussian component in the mixture distribution,
            in the order (mean, covariance, weight)
        
        sample_size : integer
            Number of samples drawn from the distribution being modelled that are
            to be used in the calculation of the loss on each learning iteration.
        
        initial_mean : float 
            Initial location of the mean of the Gaussian distribution being learnt.
            If no mean passed the intial location is set to the origin.

        dimensions : integer
            Dimensionality of the space being modelled.
            
        """
        if initial_mean is not None:
            self.mean = torch.nn.Parameter(torch.tensor(np.array(initial_mean).reshape(dimensions, 1)).float())
        else:
            self.mean = torch.nn.Parameter(torch.tensor(np.array([0] * dimensions).reshape(dimensions, 1)).float())
        self.covariance = torch.nn.Parameter(torch.eye(dimensions))
        self.dist_params = dist_params
        self.dimensions = dimensions
        self.sample_size = sample_size

    def p(self, X):
        """
        Evaulates the log probability of the sample points X under the
        mixture Gaussian distribution which is being learnt by the model.

        Parameters
        ----------
        X : torch tensor, containing sample points from the mixture
            distribution being modelled.

        Returns
        -------
        torch tensor containing the log probabilities of the sample points
        under the mixture Gaussian distribution.
        """

        total_probability = torch.zeros(self.sample_size,1)
        for params in self.dist_params:
            mean, covariance, weight = params
            # Convert params into tensor parameters:
            mean = torch.tensor(np.array([m for m in mean]).reshape(len(mean), 1)).float()
            covariance = torch.tensor(covariance).float()
            probabilities = self.normal(X, mean, covariance)
            total_probability += weight * probabilities /10000
        return total_probability

    def log_p(self, X):
        """
        Evaulates the log probability of the sample points X under the
        mixture Gaussian distribution which is being learnt by the model.

        Parameters
        ----------
        X : torch tensor, containing sample points from the mixture
            distribution being modelled.

        Returns
        -------
        torch tensor containing the log probabilities of the sample points
        under the mixture Gaussian distribution.
        """

        total_probability = torch.zeros(self.sample_size,1)
        for params in self.dist_params:
            mean, covariance, weight = params
            mean = torch.tensor(np.array([m for m in mean]).reshape(len(mean), 1)).float()
            covariance = torch.tensor(covariance).float()
            probabilities = self.normal(X, mean, covariance)
            total_probability += weight * probabilities /10000
        log_total_probability = total_probability.log()
        return log_total_probability

    def log_q(self, X):
        """
        Evaulates the log probability of the sample points in X under the
        single (multi)variate Gaussian distribution with learnt mean and
        covariance parameters.

        Parameters
        ----------
        X : torch tensor, containing sample points from the mixture
            distribution being modelled.

        Returns
        -------
        torch tensor containing the log probabilities of the sample points
        under the learnt Gaussian.
        """

        probability = self.normal(X, self.mean, self.covariance) /10000
        log_probability = probability.log()
        return log_probability

    def q(self, X):
        """
        Evaulates the probability of the sample points in X under the
        single (multi)variate Gaussian distribution with learnt mean and
        covariance parameters.

        Parameters
        ----------
        X : torch tensor, containing sample points from the mixture
            distribution being modelled.

        Returns
        -------
        torch tensor containing the probabilities of the sample points
        under the learnt Gaussian.
        """

        probability = self.normal(X, self.mean, self.covariance) /10000
        return probability
    
    def normal(self, X, m, C):
        """
        Evaluates the density of a normal distribution.

        Parameters
        ----------
        X : torch tensor, contains the data points in the
            sample being computed.

        m : torch tensor, contains the vector of means
            of the normal distribution.

        C : torch tensor, contains the variance matrix of
            the normal distribution.
        
        Returns
        -------
        torch tensor containing the probability of the points in the
        passed X tensor under the Gassian distribution described by the
        passed m and C parameters.
        """

        Z = X - m

        return torch.exp(
            - torch.sum(Z * torch.matmul(torch.inverse(C), Z), 1) / 2.
            - torch.log(torch.det(C)) / 2.
            - len(m) / 2. * torch.log(2. * torch.tensor(np.pi)))


class fwdKL(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='/home/tam63/results/synthetic-experiments',
                 **kwargs):
        super(fwdKL, self).__init__(**kwargs)

        # Logger for storing the parameter values during training:
        if not os.path.exists(f"{train_data_file_path}/fwdKL"):
            print("Creating folder!!")
            os.makedirs(f"{train_data_file_path}/fwdKL")
        elif os.path.isfile(f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}"):
            os.remove(f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}")
        file_path = f"{train_data_file_path}/fwdKL/{TRAIN_LOSSES_LOGFILE}"
        
        print(f"Logging file path: {file_path}")
        self.logger = logging.getLogger(f"losses_logger_fwdKL")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)
    
    def forward(self, X):
        return torch.mean(self.q(X) * (self.log_q(X) - self.log_p(X)))
    
    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")
        
        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")
        
        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")


class revKL(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='/home/tam63/results/synthetic-experiments',
                 **kwargs):
        super(revKL, self).__init__(**kwargs)

        # Logger for storing the parameter values during training:
        if not os.path.exists(f"{train_data_file_path}/revKL"):
            print("Creating folder!!")
            os.makedirs(f"{train_data_file_path}/revKL")
        elif os.path.isfile(f"{train_data_file_path}/revKL/{TRAIN_LOSSES_LOGFILE}"):
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
        return torch.mean(self.p(X) * (self.log_p(X) - self.log_q(X)))
    
    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")
        
        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")
        
        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")
    

class JS(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 train_data_file_path='/home/tam63/results/synthetic-experiments',
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
        loss = 0.5 * torch.sum(self.q(sample_data) * (self.log_q(sample_data) - (0.5 * self.p(sample_data) + 0.5 * self.q(sample_data)).log())) \
            + 0.5 * torch.sum(self.p(sample_data) * (self.log_p(sample_data) - (0.5 * self.p(sample_data) + 0.5 * self.q(sample_data)).log()))
        return loss
    
    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")
        
        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")
        
        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")



class GJS(BaseDivergence, torch.nn.Module):
    def __init__(self,
                 alpha=0.5,
                 dual=False,
                 train_data_file_path='/home/tam63/results/synthetic-experiments',
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
        if self.dual:
            KL_1 = self.q(sample_data).pow(self.alpha) * self.p(sample_data).pow(1 - self.alpha) * (self.p(sample_data) / self.q(sample_data)).log()
            KL_2 = self.q(sample_data).pow(self.alpha) * self.p(sample_data).pow(1 - self.alpha) * (self.q(sample_data) / self.p(sample_data)).log()
            loss = ((1 - self.alpha) ** 2) * torch.sum(KL_1) + (self.alpha ** 2) * torch.sum(KL_2)
        else:
            KL_1 = self.q(sample_data) * (self.q(sample_data) / self.p(sample_data)).log()
            KL_2 = self.p(sample_data) * (self.p(sample_data) / self.q(sample_data)).log()
            loss = ((1 - self.alpha) ** 2) * torch.sum(KL_1) + (self.alpha ** 2) * torch.sum(KL_2)
        return loss
    
    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},alpha,{self.alpha}")
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")
        
        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")
        
        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")

class GJStrainalpha(BaseDivergence,torch.nn.Module):
    def __init__(self,
                 alpha=0.5,
                 dual=False,
                 train_data_file_path='/home/tam63/results/synthetic-experiments',
                 **kwargs):
        super(GJStrainalpha, self).__init__(**kwargs)
        
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]))
        self.dual = dual

        # Logger for storing the parameter values during training:
        log_folder = f"{train_data_file_path}/{'tGJS' if dual == False else 'tdGJS'}-A_0={alpha}"
        log_file = f"{train_data_file_path}/{'tGJS' if dual == False else 'tdGJS'}-A_0={alpha}/{TRAIN_LOSSES_LOGFILE}"
        print(f"Logging file path: {log_file}")

        if not os.path.exists(log_folder):
            print("Creating folder!!")
            os.makedirs(log_folder)
        elif os.path.isfile(log_file):
            os.remove(log_file)
        
        self.logger = logging.getLogger(f"losses_logger_{'tGJS' if dual == False else 'tdGJS'}-A_0={alpha}")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)
    
    def forward(self, sample_data):
        if self.dual:
            KL_1 = self.q(sample_data).pow(self.alpha) * self.p(sample_data).pow(1 - self.alpha) * (self.p(sample_data) / self.q(sample_data)).log()
            KL_2 = self.q(sample_data).pow(self.alpha) * self.p(sample_data).pow(1 - self.alpha) * (self.q(sample_data) / self.p(sample_data)).log()
            loss = ((1 - self.alpha) ** 2) * torch.mean(KL_1) + (self.alpha ** 2) * torch.mean(KL_2)
        else:
            KL_1 = self.q(sample_data) * (self.q(sample_data) / self.p(sample_data)).log()
            KL_2 = self.p(sample_data) * (self.p(sample_data) / self.q(sample_data)).log()
            loss = ((1 - self.alpha) ** 2) * torch.mean(KL_1) + (self.alpha ** 2) * torch.mean(KL_2)
        return loss
    
    def log(self, epoch, av_divergence):
        """Write to the log file."""
        self.logger.debug(f"{epoch},alpha,{self.alpha.item()}")
        self.logger.debug(f"{epoch},av_div_loss,{av_divergence.item()}")
        
        for i, m in enumerate(self.mean.detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},mean_{i+1},{m}")
        
        for i, var in enumerate(self.covariance.diag().detach().numpy().reshape(-1)):
            self.logger.debug(f"{epoch},var_{i+1},{var}")



# Data loading and saving functions:

def get_sample_data(save_loc, seed=12334, dimensions=2, num_gaussians=5, num_samples=100000, save=False):
    """
    Definition
    ----------
    Function which produces a 2D add-mixture of Gaussian distribution, which has
    num_gaussians number of gaussian components, each of which has parameters
    radomly generated. The parameters of each component are the mean, covariance
    and weight.

    Parameters
    ----------
    seed : int
        The integer to be used to initialise the random generator.
    
    num_gaussians : int
        Number of Gaussian components to be used in the mxture distribution.

    num_samples : int
        Number of samples to be taken from the generated mixture distribution
        and returned in the data_samples vector.
    
    save : bool
        Whether to save a plot of the generated mixture distribution.
    
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

    p = np.random.dirichlet([.5] * num_gaussians)
    v = 1. / np.square(np.random.rand(num_gaussians) + 1.)
    C = [np.eye(dimensions) * _ for _ in v]

    dist_params = [[[np.random.rand()*6 - 3 for i in range(dimensions)], covariance, weight] for covariance, weight in zip(C, p)]
    data_samples = np.concatenate([multivariate_normal.rvs(mean=m, cov=covariance, size=max(int(num_samples*weight), 2)) for m, covariance, weight in dist_params])

    if save and dimensions == 2:
        x, y = np.mgrid[-6:6:.01, -6:6:.01]
        pos = np.dstack((x, y))
        pdf = np.array([weight * multivariate_normal(mean, covariance).pdf(pos) for mean, covariance, weight in dist_params])

        fig = plt.figure(figsize=(10,10))
        ax2 = fig.add_subplot(111)
        ax2.contour(x, y, pdf.sum(axis=0)/pdf.sum(axis=0).sum())
        plt.scatter(data_samples[:,0], data_samples[:,1], s=0.01, c ='k')
        if not os.path.exists(f"{save_loc}"):
            os.makedirs(f"{save_loc}")
        plt.axis('equal')
        plt.axis([np.min(data_samples[:,0]), np.max(data_samples[:,0]), np.min(data_samples[:,1]), np.max(data_samples[:,1])])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().tight_layout()
        plt.savefig(f"{save_loc}/mixture-distribution.png", dpi=200)

    return dist_params, data_samples


def train_model(model, optimizer, data_samples, log_loc, learn_a=False, dimensions=2, epochs=20, sample_size=200, plot_learnt_dist=False, name=None, save_loc=None, frequency=2):
    """
    Description
    -----------
    Function that trains a passed model to learn the parameters of a
    multivariate Gaussian distribution to approximate a mixture multivariate
    Gaussian.
    """
    start = default_timer()
    for e in range(epochs):
        divergence = 0
        np.random.shuffle(data_samples)
        print(f"Epoch {e + 1}")
        for i in tqdm(range(int(len(data_samples) / sample_size))):

            sample = torch.tensor(data_samples[i : i + sample_size]).view(sample_size, dimensions, 1).float()
            
            loss = model(sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            divergence += loss.detach()
            
            with torch.no_grad():
                if learn_a:
                    model.alpha.clamp_(0, 1)
                
                # Enforce the covariance matrix to diagonal:
                for i in range(dimensions):
                    for j in range(dimensions):
                        if i == j:
                            continue
                        else:
                            model.covariance[i][j] = 0.0
        model.log(e, divergence/int(len(data_samples) / sample_size))

    dt = (default_timer() - start) / 60
    print(f'Finished training after {dt:.1f} min.')

    # Plotting learnt distribution:
    if (plot_learnt_dist) and (save_loc != None) and (name != None) and (dimensions == 2):
        x, y = np.mgrid[-6:6:.01, -6:6:.01]
        pos = np.dstack((x, y))
        pdf = multivariate_normal(model.mean.detach().numpy().reshape(-1), model.covariance.detach().numpy()).pdf(pos)
        fig = plt.figure(figsize=(10,10))
        ax2 = fig.add_subplot(111)
        ax2.contour(x, y, pdf)
        plt.scatter(data_samples[:,0], data_samples[:,1], s=0.01, c='k')
        plt.axis('equal')
        plt.axis([np.min(data_samples[:,0]), np.max(data_samples[:,0]), np.min(data_samples[:,1]), np.max(data_samples[:,1])])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().tight_layout()
        plt.savefig(f"{save_loc}/fitted-dist-{name}.png", dpi=200)
    
    # Save learnt parameters:
    learnt_dist = {}
    if learn_a:
        learnt_dist[f"alpha"] = str(model.alpha.item())
    elif hasattr(model, 'alpha'):
            learnt_dist[f"alpha"] = str(model.alpha)
    for i, m in enumerate(model.mean.detach().numpy().reshape(-1)):
        learnt_dist[f"mean_{i + 1}"] = str(m)
    for i, var in enumerate(model.covariance.diag().detach().numpy().reshape(-1)):
        learnt_dist[f"var_{i+1}"] = str(var)
    with open(f'{log_loc}/{name}/final-parameters.txt', 'w') as f:
        json.dump(learnt_dist, f)

    return model

    
def main(argv):
    parser = ArgumentParser(argv[0],
                            description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--metrics', '-m', choices=['fwdKL', 'revKL', 'JS', 'GJS', 'dGJS', 'tGJS', 'tdGJS'], nargs='+',
                        help='Which metrics to include in comparison.')
    parser.add_argument('--num-data', '-N', type=int, default=100000,
                        help='Number of points in the sampled data from the generated mixture distribution.')
    parser.add_argument('--sample-size', '-S', type=int, default=200,
                        help='Number of points in the each training batch in which the gradient is calculated.')
    parser.add_argument('--num-mixture', type=int, default=5,
                        help='Number of Gaussian components in the mixture distribution.')
    parser.add_argument('--dimensions', type=int, default=2,
                        help='Dimensions of the Gaussian distributions being modelled.')
    parser.add_argument('--seed', '-s', type=int, default=22,
                        help='Random seed used to generate data.')
    parser.add_argument('--A0', type=float, default=0.5,
                        help='Initial value of alpha if training GJS or dGJS.')
    parser.add_argument('--lr', '--learning-rate',
                        type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--plot-output', type=str, default='/home/tam63/figures/synthetic-experiments',
                        help='Where to store plots produced.')
    parser.add_argument('--train-log-output', '-o', type=str, default='/home/tam63/results/synthetic-experiments',
                        help='Where to store log of data produced during training.')
    parser.add_argument('--save-mixture-plot', type=bool, default=True,
                        help='Where to store results.')


    args = parser.parse_args(argv[1:])

    plot_loc = f"{args.plot_output}/{args.seed}-{args.dimensions}-{args.num_mixture}-{args.lr}-{args.epochs}"
    log_loc = f"{args.train_log_output}/{args.seed}-{args.dimensions}-{args.num_mixture}-{args.lr}-{args.epochs}"

    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)
    if not os.path.exists(log_loc):
        os.makedirs(log_loc)
    
    print('Generating data...')
    dist_params, data_samples = get_sample_data(dimensions=args.dimensions, save_loc=plot_loc, save=True, seed=args.seed, num_gaussians=args.num_mixture, num_samples=args.num_data)
    
    if 'tGJS' in args.metrics:
        print('Optimizing trainable-alpha Geometric Jensen-Shannon divergence...')
        model = GJStrainalpha(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, dual=False, train_data_file_path=log_loc, alpha=args.A0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, learn_a=True, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'tGJS-A_0={args.A0}', save_loc=plot_loc)
    
    if 'tdGJS' in args.metrics:
        print('Optimizing trainable-alpha dual Geometric Jensen-Shannon divergence...')
        model = GJStrainalpha(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, dual=True, train_data_file_path=log_loc, alpha=args.A0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, learn_a=True, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                  plot_learnt_dist=True, name=f'tdGJS-A_0={args.A0}', save_loc=plot_loc)
    
    if 'GJS' in args.metrics:
        print('Optimizing constant-alpha Geometric Jensen-Shannon divergence...')
        model = GJS(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, dual=False, train_data_file_path=log_loc, alpha=args.A0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'GJS-A_0={args.A0}', save_loc=plot_loc)
    
    if 'dGJS' in args.metrics:
        print('Optimizing constant-alpha dual Geometric Jensen-Shannon divergence...')
        model = GJS(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, dual=True, train_data_file_path=log_loc, alpha=args.A0)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'dGJS-A_0={args.A0}', save_loc=plot_loc)

    if 'JS' in args.metrics:
        print('Optimizing Jensen-Shannon divergence...')
        model = JS(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, train_data_file_path=log_loc)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'JS', save_loc=plot_loc)    

    if 'revKL' in args.metrics:
        print('Optimizing Reverse Kullback-Leibler divergence...')
        model = revKL(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, train_data_file_path=log_loc)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'revKL', save_loc=plot_loc)
    
    if 'fwdKL' in args.metrics:
        print('Optimizing Kullback-Leibler  divergence...')
        model = fwdKL(dist_params=dist_params, dimensions=args.dimensions, sample_size=args.sample_size, train_data_file_path=log_loc)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = train_model(model, optimizer, data_samples, log_loc=log_loc, dimensions=args.dimensions, epochs=args.epochs, sample_size=args.sample_size,
                                 plot_learnt_dist=True, name=f'fwdKL', save_loc=plot_loc)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))