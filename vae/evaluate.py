"""Evaluation module."""

import logging
import math
import os
import torch

from collections import defaultdict
from logging import Logger
from functools import reduce
from timeit import default_timer
from tqdm import tqdm, trange
from typing import Any, Dict, Optional, Tuple

from vae.utils.math import log_density_gaussian
from vae.utils.modelIO import save_metadata


TRAIN_EVAL_FILE = "train_eval.log"
TEST_EVAL_FILE = "test_eval.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"


class Evaluator():

    def __init__(self,
                 model: Any,
                 loss_f: Any,
                 device: Optional[torch.device] = torch.device("cpu"),
                 is_metrics: Optional[bool] = False,
                 is_train: Optional[bool] = False,
                 logger: Optional[Logger] = logging.getLogger(__name__),
                 save_dir: Optional[str] = "results",
                 is_progress_bar: Optional[bool] = True,
                 denoise: Optional[bool] = False
                 ) -> None:
        self.model = model.to(device)
        self.loss_f = loss_f
        self.device = device
        self.is_metrics = is_metrics
        self.is_train = is_train
        self.logger = logger
        self.logger.info(f"Testing Device: {self.device}")
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.denoise = denoise

    def __call__(self,
                 data_loader: torch.utils.data.DataLoader
                 ) -> Tuple[Dict, Dict]:
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metrics = None
        if self.is_metrics:
            self.logger.info('Computing metrics...')
            metrics, metric_helpers = self.compute_metrics(data_loader)
            self.logger.info(f'Metrics: {metrics}')
            self.logger.info(f'Metrics: {metric_helpers}')
            save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)

        self.logger.info('Computing losses...')
        losses = self.compute_losses(data_loader)
        self.logger.info(f'Losses: {losses}')
        if self.is_train:
            save_metadata(losses, self.save_dir, filename=TRAIN_EVAL_FILE)
        else:
            save_metadata(losses, self.save_dir, filename=TEST_EVAL_FILE)

        if is_still_training:
            self.model.train()

        dt = (default_timer() - start) / 60
        self.logger.info(f'Finished evaluating after {dt:.1f} min.')

        return metrics, losses

    def compute_losses(self, data_loader: torch.utils.data.DataLoader):
        storer = defaultdict(list)
        total_rec_loss = 0.
        if self.denoise:
            for noise_data, clean_data in tqdm(
                    data_loader, leave=False,
                    disable=not self.is_progress_bar):
                noise_data = noise_data.to(self.device)
                clean_data = clean_data.to(self.device)

                recon_batch, latent_dist, latent_sample = self.model(noise_data)
                loss, rec_loss = self.loss_f(
                    clean_data, recon_batch, latent_dist, self.model.training,
                    storer, latent_sample=latent_sample)
                total_rec_loss += rec_loss.item()
        else:
            for data, _ in tqdm(data_loader,
                                leave=False,
                                disable=not self.is_progress_bar):
                data = data.to(self.device)

                recon_batch, latent_dist, latent_sample = self.model(data)
                loss, rec_loss = self.loss_f(
                    data, recon_batch, latent_dist, self.model.training,
                    storer, latent_sample=latent_sample)
                total_rec_loss += rec_loss.item()

        losses = {k: sum(v) / len(data_loader) for k, v in storer.items()}
        losses['recon_loss'] = total_rec_loss / len(data_loader)

        return losses

    def compute_metrics(self, data_loader):
        """
        Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        try:
            lat_sizes = data_loader.dataset.lat_sizes
            lat_names = data_loader.dataset.lat_names
        except AttributeError:
            raise ValueError(
                "Dataset needs to have known true factors of variations to" +
                "compute the metric. This does not seem to be the case for" +
                f" {type(data_loader.__dict__['dataset']).__name__}")

        self.logger.info("Computing the empirical distribution q(z|x)...")
        samples_zCx, params_zCx = self._compute_q_zCx(data_loader)
        len_dataset, latent_dim = samples_zCx.shape

        self.logger.info("Estimating marginal entropies...")
        # marginal entropy H(z_j)
        H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)

        # conditional entropy H(z|v)
        samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
        params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
        H_zCv = self._estimate_H_zCv(
            samples_zCx, params_zCx, lat_sizes, lat_names)

        H_z = H_z.cpu()
        H_zCv = H_zCv.cpu()

        # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j]
        #            = - H[z_j|v_k] + H[z_j]
        mut_info = - H_zCv + H_z
        sorted_mut_info = \
            torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)

        metric_helpers = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
        mig = self._mutual_information_gap(
            sorted_mut_info, lat_sizes, storer=metric_helpers)

        metrics = {'MIG': mig.item()}
        torch.save(
            metric_helpers, os.path.join(self.save_dir, METRIC_HELPERS_FILE))

        return metrics, metric_helpers

    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """
        Compute the mutual information gap as in [1].

        References
        ----------
            [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in
            variational autoencoders." Advances in Neural Information
            Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of
        # variation, then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _compute_q_zCx(self, dataloader):
        """
        Compute the empirical disitribution of q(z|x).

        Parameter
        ---------
        dataloader: torch.utils.data.DataLoader
            Batch data iterator.

        Return
        ------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
        """
        len_dataset = len(dataloader.dataset)
        latent_dim = self.model.latent_dim
        n_suff_stat = 2

        q_zCx = torch.zeros(
            len_dataset, latent_dim, n_suff_stat, device=self.device)

        n = 0
        with torch.no_grad():
            for x, label in dataloader:
                batch_size = x.size(0)
                idcs = slice(n, n + batch_size)
                q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = \
                    self.model.encoder(x.to(self.device))
                n += batch_size

        params_zCX = q_zCx.unbind(-1)
        samples_zCx = self.model.reparameterize(*params_zCX)

        return samples_zCx, params_zCX

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
        r"""
        Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)]
                               = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
        using the emperical distribution of :math:`p(x)`.

        Note
        ----
        - the expectation over the emperical distribution is:
          :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
        - assume that q(z|x) is factorial i.e.
          :math:`q(z|x) = \prod_j q(z_j|x)`.
        - computes numerically stable NLL:
          :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

        Parameters
        ----------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

        Return
        ------
        H_z: torch.Tensor
            Tensor of shape (latent_dim) - the marginal entropies H(z_j).
        """
        len_dataset, latent_dim = samples_zCx.shape
        device = samples_zCx.device
        H_z = torch.zeros(latent_dim, device=device)

        # sample from p(x)
        samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
        # sample from p(z|x)
        samples_zCx = \
            samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = params_zCX[0].unsqueeze(-1).expand(
            len_dataset, latent_dim, n_samples)
        log_var = params_zCX[1].unsqueeze(-1).expand(
            len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)
        with trange(
                n_samples, leave=False, disable=not self.is_progress_bar) as t:
            for k in range(0, n_samples, mini_batch_size):
                # log q(z_j|x) for n_samples
                idcs = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                 mean[..., idcs],
                                                 log_var[..., idcs])
                # numerically stable log q(z_j) for n_samples:
                # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
                # As we don't know q(z) we appoximate it with the monte carlo
                # expectation of q(z_j|x_n) over x. => fix single z and look at
                # proba for every x to generate it. n_samples is not used here!
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0)
                # H(z_j) = E_{z_j}[- log q(z_j)]
                # mean over n_samples (dim 1 because already summed over 0).
                H_z += (-log_q_z).sum(1)

                t.update(mini_batch_size)

        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """
        Estimate conditional entropies :math:`H[z|v]`.
        """
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes)
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device)
        for i, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for j in range(lat_size):
                self.logger.info(
                    f"Estimating conditional entropies for the {j}th value " +
                    f"of {lat_name}.")
                idcs[i] = j
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(
                    len_dataset // lat_size, latent_dim)
                params_zxCv = tuple(
                    p[idcs].contiguous().view(
                        len_dataset // lat_size, latent_dim)
                    for p in params_zCx)

                H_zCv[i] += self._estimate_latent_entropies(
                    samples_zxCv, params_zxCv) / lat_size

        return H_zCv
