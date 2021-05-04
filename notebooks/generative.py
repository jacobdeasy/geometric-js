import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from tqdm import tqdm
import argparse

import pandas as pd
import sys

BASE_DIR=os.path.dirname(os.getcwd())
sys.path.append(BASE_DIR)
sys.path.append('/home/tam63/geometric-js')
import torch

import scipy.stats
from scipy.stats import norm
from scipy.special import logsumexp

from vae.utils.modelIO import save_model, load_model, load_metadata
from notebooks.utils import PlotParams
# from utils.helpers import (create_safe_directory, get_device, set_seed,
#                            get_n_param)


TRAIN_MODELS_DIR = "/home/tam63/results/alpha-experiments"
DATA_DIR = "/home/tam63/geometric-js/data"
SAVE_DIR = "/home/tam63/figures/alpha-experiments"


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    description = "PyTorch implementation and evaluation of Variational" + \
                    "AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('--dataset', type=str, choices=['mnist', 'fashion', 'dsprites'],
                        help="Name of the dataset being plotted.")
    general.add_argument('--divergence', type=str, choices=['dGJS', 'GJS', 'both'],
                        help="Type of geometric-JS divergence to be plotted on comparison plot.")
    general.add_argument('--model-loc', type=str,
                        help="Location of the trained models to be used to generate plots.")

    args = parser.parse_args(args_to_parse)
    print(args)

    return args


def bootstrap(x, low, high, n_samples):
    mu = x.mean()
    n = len(x)
    X = np.random.choice(x, size=n_samples*n).reshape(n_samples, n)
    mu_star = X.mean(axis=1)
    d_star = np.sort(mu_star - mu)

    return mu, mu + d_star[int(low*n_samples)], mu + d_star[int(high*n_samples)]


def compute_samples(model, data, num_samples, debug=False):
    """ 
        Description
        ---------------------------------------------------------------

        Sample from importance distribution z_samples ~ q(z|X) and
        compute p(z_samples), q(z_samples) for importance sampling


        Inputs
        ---------------------------------------------------------------
        model : pytorch nn.Module

                VAE model implemented in pytroch which has been
                trained on the training data corresponding to the
                passed test data, which is contained in the variable
                'data'.
        
        data : pytorch Tensor

               Tensor of shape [batch_size, 1, im_size, im_size], 
               where im_size is the dimension size of the images used
               to train the model, and batch size is the number of
               data instances passed, which is therefore also the
               number of estimates of the probability distribution
               which will be produced.

        num_samples : int

               For each passed data instance, the probability
               distribution p(x|z) will be estimated using a monte
               carlo integration with num_samples samples.
        

        returns
        ---------------------------------------------------------------

        z_samples, pz, qz : numpy array

              Returns arrays containing the representation of each
              passed input image in latent space in z_samples, and the
              probabilty distributions qz and pz which are defined by
              samples drawn from the normal distribution defined by the
              latent space (qz) and defined by the latent space


    """
    data = data.cuda()
    z_mean, z_log_sigma = model.encoder(data)
    z_mean = z_mean.cpu().detach().numpy()
    z_log_sigma = z_log_sigma.cpu().detach().numpy()
    z_samples = []
    qz = []

    for m, s in zip(z_mean, z_log_sigma):
        # len(s) = len(s) = 10 = size of the latent space dimension
        #
        # z_vals is num_samples (= 128) samples drawn from the normal
        # distribution defined by the mean and std (m[i], s[i])
        #
        # qz_vals is the normal distribution defined by the samples
        # in the vector z_vals
      
        z_vals = [np.random.normal(m[i], np.exp(s[i]), num_samples) for i in range(len(m))]
        qz_vals = [norm.pdf(z_vals[i], loc=m[i], scale=np.exp(s[i])) for i in range(len(m))]
        z_samples.append(z_vals)
        qz.append(qz_vals)

    z_samples = np.array(z_samples)
    pz = norm.pdf(z_samples)
    qz = np.array(qz)
    # pdb.set_trace()
    # Check why the axes are being swapped

    z_samples = np.swapaxes(z_samples, 1, 2)
    pz = np.swapaxes(pz, 1, 2)
    qz = np.swapaxes(qz, 1, 2)

    return z_samples, pz, qz

def estimate_logpx_batch(model, data, num_samples, debug=False, digit_size=32):
    """
  
    """
    z_samples, pz, qz = compute_samples(model, data, num_samples)
    assert len(z_samples) == len(data)
    assert len(z_samples) == len(pz)
    assert len(z_samples) == len(qz)
    z_samples = torch.tensor(z_samples).float().cuda()

    result = []
    for i in range(len(data)):
        x_predict = model.decoder(z_samples[i]).reshape(-1, digit_size ** 2)
        x_predict = x_predict.cpu().detach().numpy()
        x_predict = np.clip(x_predict, np.finfo(float).eps, 1. - np.finfo(float).eps)
        p_vals = pz[i]
        q_vals = qz[i]

        # pdb.set_trace()
        datum = data[i].cpu().reshape(digit_size ** 2).numpy() #.reshape(digit_size ** 2)
        
        # \log p(x|z) = Binary cross entropy
        logp_xz = np.sum(datum * np.log(x_predict + 1e-9) + (1. - datum) * np.log(1.0 - x_predict + 1e-9), axis=-1)
        logpz = np.sum(np.log(p_vals + 1e-9), axis=-1)
        logqz = np.sum(np.log(q_vals + 1e-9), axis=-1)
        argsum = logp_xz + logpz - logqz
        logpx = -np.log(num_samples + 1e-9) + logsumexp(argsum)
        result.append(logpx)

    return np.array(result)

def estimate_logpx(model, data, num_samples, verbosity=0, digit_size=32):
    batches = []
    iterations = int(np.ceil(1. * len(data) / 100))
    for b in tqdm(range(iterations)):
        batch_data = data[b * 100:(b + 1) * 100]
        batches.append(estimate_logpx_batch(model, batch_data, num_samples, digit_size=digit_size))
        if verbosity and b % max(11 - verbosity, 1) == 0:
            print("Batch %d [%d, %d): %.2f" % (b, b * 100, (b+1) * 100, np.mean(np.concatenate(batches))))

    log_probs = np.concatenate(batches)
    mu, lb, ub = bootstrap(log_probs, 0.025, 0.975, 1000)

    return mu, lb, ub

def main(args):
    device = 'cuda'
    plotter = PlotParams()
    plotter.set_params()
    DATA_DIR = os.path.join(os.pardir, 'data')
    FIG_DIR = os.path.join(os.pardir, 'figs')
    RES_DIR = os.path.join(os.pardir, 'results')

    # 1) select dataset to load:

    if args.dataset == 'dsprites':
        X_test = np.load(os.path.join(DATA_DIR, 'dsprites', 'dsprite_train.npz'))['imgs']
        X_test = torch.tensor(X_test).unsqueeze(1).float() / 255.0
        digit_size = 64
        X_test = X_test[:10000]
        X_test = X_test.to(device)

    elif args.dataset == 'fashion':
        X_test = torch.load(os.path.join(DATA_DIR, 'fashionMnist', 'FashionMNIST', 'processed', 'test.pt'))
        digit_size = 32
        X_test = X_test[0].unsqueeze(1).float() / 255.0
        X_test = torch.nn.functional.pad(X_test, pad=(2, 2, 2, 2))
        X_test = X_test[:10000]
        X_test = X_test.to(device)

    elif args.dataset == 'mnist':
        X_test = torch.load(os.path.join(DATA_DIR, 'mnist', 'MNIST', 'processed', 'test.pt'))
        digit_size = 32
        X_test = X_test[0].unsqueeze(1).float() / 255.0
        X_test = torch.nn.functional.pad(X_test, pad=(2, 2, 2, 2))
        X_test = X_test[:10000]
        X_test = X_test.to(device)


    # 2) Get the trained alpha dGJS probabilities:

    av_a = []
    log_probs_lb = []
    log_probs_ub = []
    log_probs_mu = []
    log_probs_best = -np.inf

    if args.divergence in ['GJS', 'dGJS']:
        divergence = args.divergence
        for initial_a in [i/10 for i in range(11)]:
            
            model_path = f"{TRAIN_MODELS_DIR}/{args.dataset}/{args.model_loc}/{divergence}-A_0={initial_a}"
            model = load_model(model_path)
            logpx_mu, logpx_lb, logpx_ub = estimate_logpx(model, X_test, num_samples=128, verbosity=0, digit_size=digit_size)
            
            log_probs_mu += [logpx_mu]
            log_probs_lb += [logpx_lb]
            log_probs_ub += [logpx_ub]
            
            if logpx_mu > log_probs_best:
                model_best = model_path
                log_probs_best = logpx_mu
            # break
            print(model_path)
            print("log p(x) = %.2f (%.2f, %.2f)" % (logpx_mu, logpx_lb, logpx_ub))
            

    # 3) Get the comparison divergences probabilities:
    av_a_i = []
    log_probs_lb_i = []
    log_probs_ub_i = []
    log_probs_mu_i = []
    log_probs_best_i = -np.inf
    model_names = []


    # KL:
    model_path = f"{TRAIN_MODELS_DIR}/{args.dataset}/{args.model_loc}/KL"
    model = load_model(model_path)
    logpx_mu, logpx_lb, logpx_ub = estimate_logpx(model, X_test, num_samples=128, verbosity=0, digit_size=digit_size)

    log_probs_mu_i += [logpx_mu]
    log_probs_lb_i += [logpx_lb]
    log_probs_ub_i += [logpx_ub]
    model_names.append("KL")

    # break
    print(model_path)
    print("log p(x) = %.2f (%.2f, %.2f)" % (logpx_mu, logpx_lb, logpx_ub))


    # fwdKL:
    model_path = f"{TRAIN_MODELS_DIR}/{args.dataset}/{args.model_loc}/fwdKL"
    model = load_model(model_path)
    logpx_mu, logpx_lb, logpx_ub = estimate_logpx(model, X_test, num_samples=128, verbosity=0, digit_size=digit_size)

    log_probs_mu_i += [logpx_mu]
    log_probs_lb_i += [logpx_lb]
    log_probs_ub_i += [logpx_ub]
    model_names.append("fwdKL")

    # break
    print(model_path)
    print("log p(x) = %.2f (%.2f, %.2f)" % (logpx_mu, logpx_lb, logpx_ub))


    # MMD:
    model_path = f"{TRAIN_MODELS_DIR}/{args.dataset}/{args.model_loc}/MMD"
    model = load_model(model_path)
    logpx_mu, logpx_lb, logpx_ub = estimate_logpx(model, X_test, num_samples=128, verbosity=0, digit_size=digit_size)

    log_probs_mu_i += [logpx_mu]
    log_probs_lb_i += [logpx_lb]
    log_probs_ub_i += [logpx_ub]
    model_names.append("MMD")

    # break
    print(model_path)
    print("log p(x) = %.2f (%.2f, %.2f)" % (logpx_mu, logpx_lb, logpx_ub))


    # no-constraint:
    # model_path = f"{TRAIN_MODELS_DIR}/{args.dataset}/{args.model_loc}/no-constraint"
    # model = load_model(model_path)
    # logpx_mu, logpx_lb, logpx_ub = estimate_logpx(model, X_test, num_samples=128, verbosity=0, digit_size=digit_size)

    # log_probs_mu_i += [logpx_mu]
    # log_probs_lb_i += [logpx_lb]
    # log_probs_ub_i += [logpx_ub]
    # model_names.append("no-constraint")

    # print(model_path)
    # print("log p(x) = %.2f (%.2f, %.2f)" % (logpx_mu, logpx_lb, logpx_ub))
    

    # 4) Plot:


    fig = plt.figure(figsize=(10, 10))

    yerr_bar = np.array(log_probs_ub) - np.array(log_probs_lb)
    yerr_bar_i = np.array(log_probs_ub_i) - np.array(log_probs_lb_i)
    initial_a = [i/10 for i in range(11)]
    plt.errorbar(initial_a, log_probs_mu, yerr=yerr_bar, label=args.divergence)

    for i in range(len(model_names)):
        plt.errorbar(initial_a, [log_probs_mu_i[i]] * len(initial_a), yerr=[yerr_bar_i[i]] * len(initial_a), label=model_names[i])
    plt.xlabel(r'Initial $\alpha$')
    plt.ylabel(r'$\log(p_{\theta}(X))$')
    plt.legend()
    plt.title("Log model evidence vs initial alpha")
    plt.savefig(f"{SAVE_DIR}/{args.dataset}/{args.divergence}/{args.divergence}-generative-performance.pdf")
    plt.savefig(f"{SAVE_DIR}/{args.dataset}/{args.divergence}/{args.divergence}-generative-performance.png", dpi=200)



    # save tight layout version:
    fig = plt.figure(figsize=(10, 10))

    yerr_bar = np.array(log_probs_ub) - np.array(log_probs_lb)
    yerr_bar_i = np.array(log_probs_ub_i) - np.array(log_probs_lb_i)
    initial_a = [i/10 for i in range(11)]
    plt.errorbar(initial_a, log_probs_mu, yerr=yerr_bar, label=args.divergence)

    for i in range(len(model_names)):
        plt.errorbar(initial_a, [log_probs_mu_i[i]] * len(initial_a), yerr=[yerr_bar_i[i]] * len(initial_a), label=model_names[i])
    plt.xlabel(r'Initial $\alpha$')
    plt.ylabel(r'$\log(p_{\theta}(X))$')
    plt.legend()
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.savefig(f"{SAVE_DIR}/{args.dataset}/{args.divergence}/{args.divergence}-generative-performance-tight-layout.pdf")
    plt.savefig(f"{SAVE_DIR}/{args.dataset}/{args.divergence}/{args.divergence}-generative-performance-tight-layout.png", dpi=200)

    

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)