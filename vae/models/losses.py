"""Module containing all vae losses."""

import abc
import torch

from torch import optim, Tensor
from torch.nn import functional as F
from typing import Any, Dict, Optional, Tuple

from .discriminator import Discriminator
from vae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                            matrix_log_density_gaussian)


LOSSES = ["VAE", "KL", "fwdKL", "GJS", "dGJS",
          "tGJS", "tdGJS", "MMD", "betaH", "betaB",
          "factor", "btcvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "VAE":
        return BetaHLoss(beta=0, **kwargs_all)
    elif loss_name == "KL":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "fwdKL":
        return BetaHLoss(beta=1, fwd_kl=True, **kwargs_all)
    elif loss_name == "GJS":
        return GeometricJSLoss(alpha=kwargs_parse["GJS_A"],
                               beta=kwargs_parse["GJS_B"],
                               dual=False,
                               invert_alpha=kwargs_parse["GJS_invA"],
                               **kwargs_all)
    elif loss_name == "dGJS":
        return GeometricJSLoss(alpha=kwargs_parse["GJS_A"],
                               beta=kwargs_parse["GJS_B"],
                               dual=True,
                               invert_alpha=kwargs_parse["GJS_invA"],
                               **kwargs_all)
    elif loss_name == "tGJS":
        return GeometricJSLossTrainableAlpha(alpha=kwargs_parse["GJS_A"],
                                             beta=kwargs_parse["GJS_B"],
                                             dual=False,
                                             invert_alpha=kwargs_parse["GJS_invA"],
                                             **kwargs_all)
    elif loss_name == "tdGJS":
        return GeometricJSLossTrainableAlpha(alpha=kwargs_parse["GJS_A"],
                                             beta=kwargs_parse["GJS_B"],
                                             dual=True,
                                             invert_alpha=kwargs_parse["GJS_invA"],
                                             **kwargs_all)
    elif loss_name == "MMD":
        return MMDLoss(beta=kwargs_parse["MMD_B"],
                       **kwargs_all)
    elif loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           disc_kwargs=dict(
                               latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(
                               lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)),
                           **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)
    else:
        assert loss_name not in LOSSES
        raise ValueError("Unknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    r"""
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Loss record frequency.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: int, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self,
                 record_loss_every: Optional[int] = 938,
                 rec_dist: Optional[str] = "bernoulli",
                 steps_anneal: Optional[int] = 0,
                 **kwargs: Optional[Any]
                 ) -> None:
        super().__init__(**kwargs)
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self,
                 data: torch.Tensor,
                 recon_data: torch.Tensor,
                 latent_dist: Tuple[torch.Tensor, torch.Tensor],
                 is_train: bool,
                 storer: Dict,
                 **kwargs: Optional[Any]
                 ) -> None:
        r"""
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def _pre_call(self,
                  is_train: bool,
                  storer: Dict
                  ) -> Dict:
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 0:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    r"""Compute the Beta-VAE loss as in [1]

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts
        with a constrained variational framework." (2016).
    """

    def __init__(self,
                 beta: Optional[float] = 4.0,
                 fwd_kl: Optional[bool] = False,
                 **kwargs: Optional[Dict]
                 ) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.fwd_kl = fwd_kl

    def __call__(self,
                 data: torch.Tensor,
                 recon_data: torch.Tensor,
                 latent_dist: Tuple[torch.Tensor, torch.Tensor],
                 is_train: bool,
                 storer: Dict,
                 **kwargs: Optional[Any]
                 ) -> None:
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        mean, logvar = latent_dist

        if self.fwd_kl:
            kl_loss = _kl_normal_loss(
                m_1=torch.zeros_like(mean),
                lv_1=torch.zeros_like(logvar),
                m_2=mean,
                lv_2=logvar,
                storer=storer)
        else:
            kl_loss = _kl_normal_loss(
                m_1=mean,
                lv_1=logvar,
                m_2=torch.zeros_like(mean),
                lv_2=torch.zeros_like(logvar),
                storer=storer)

        loss = rec_loss + self.beta * kl_loss

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss, rec_loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer,
                 **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(
            self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
            if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # factor-vae split data into 2 batches. In paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If softmax not logisitc regression
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (bad results) should be
        # `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(
            0, 1, self.n_train_steps, self.steps_anneal)
            if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(
            half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) +
        #                    self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check if should also anneal discriminator
        # if not becomes too good ???
        # d_tc_loss = anneal_reg * d_tc_loss

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in
       variational autoencoders." Advances in Neural Information Processing
       Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)
        # batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(
            latent_sample, latent_dist, self.n_data, is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(
            0, 1, self.n_train_steps, self.steps_anneal)
            if is_train else 1)

        # total loss
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss


class MMDLoss(BaseLoss):
    r"""Compute VAE loss with maximum mean discrepancy regularisation.

    Parameters
    ----------
    beta : float, optional
        Weight of the MMD divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.
    """

    def __init__(self, beta=500.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer,
                 **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        mmd_loss = _mmd_loss(*latent_dist, storer=storer)
        loss = rec_loss + self.beta * mmd_loss

        if storer is not None:
            storer['loss'] += [loss.item()]

        return loss, rec_loss


class GeometricJSLoss(BaseLoss):
    r"""Compute VAE loss with skew geometric-Jensen-Shannon regularisation [1].

    Parameters
    ----------
    alpha : float, optional
        Skew of the skew geometric-Jensen-Shannon divergence

    beta : float, optional
        Weight of the skew g-js divergence.

    dual : bool, optional
        Whether to use Dual or standard GJS.

    invert_alpha : bool, optional
        Whether to replace alpha with 1 - a.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.

    References
    ----------
        [1] Deasy, Jacob, Nikola Simidjievski, and Pietro Liò.
        "Constraining Variational Inference with Geometric Jensen-Shannon Divergence."
        Advances in Neural Information Processing Systems 33 (2020).
    """

    def __init__(self, alpha=0.5, beta=1.0, dual=True, invert_alpha=True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.dual = dual
        self.invert_alpha = invert_alpha

    def __call__(self, data, recon_data, latent_dist, is_train, storer,
                 **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        gjs_loss = _gjs_normal_loss(*latent_dist,
                                    dual=self.dual,
                                    a=self.alpha,
                                    invert_alpha=self.invert_alpha,
                                    storer=storer)
        loss = rec_loss + self.beta * gjs_loss

        if storer is not None:
            storer['loss'] += [loss.item()]

        return loss, rec_loss


class GeometricJSLossTrainableAlpha(BaseLoss, torch.nn.Module):
    r"""Compute VAE loss with skew geometric-Jensen-Shannon regularisation [1].

    References
    ----------
        [1] Deasy, Jacob, Nikola Simidjievski, and Pietro Liò.
        "Constraining Variational Inference with Geometric Jensen-Shannon
        Divergence." Advances in Neural Information Processing Systems 33
        (2020).
    """

    def __init__(self,
                 alpha: Optional[float] = 0.5,
                 beta: Optional[float] = 1.0,
                 dual: Optional[bool] = True,
                 invert_alpha: Optional[bool] = True,
                 device: Optional[torch.device] = None,
                 **kwargs: Optional[Dict]
                 ) -> None:
        super(GeometricJSLossTrainableAlpha, self).__init__(**kwargs)

        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = beta
        self.dual = dual
        self.invert_alpha = invert_alpha
        self.device = device

    def __call__(self,
                 data,
                 recon_data,
                 latent_dist,
                 is_train,
                 storer,
                 record_alpha_range=False,
                 **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        gjs_loss = _gjs_normal_loss(*latent_dist,
                                    dual=self.dual,
                                    a=self.alpha,
                                    invert_alpha=self.invert_alpha,
                                    storer=storer,
                                    record_alpha_range=record_alpha_range)
        loss = rec_loss + self.beta * gjs_loss

        if storer is not None:
            storer['loss'] += [loss.item()]

        return loss, rec_loss


class GeometricJSLossTrainablePrior(BaseLoss, torch.nn.Module):
    r"""Compute VAE loss with skew geometric-Jensen-Shannon regularisation [1].

    Parameters
    ----------
    alpha : float, optional
        Skew of the skew geometric-Jensen-Shannon divergence

    beta : float, optional
        Weight of the skew g-js divergence.

    dual : bool, optional
        Whether to use Dual or standard GJS.

    invert_alpha : bool, optional
        Whether to replace alpha with 1 - a.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. `rec_dist`.

    References
    ----------
        [1] Deasy, Jacob, Nikola Simidjievski, and Pietro Liò.
        "Constraining Variational Inference with Geometric Jensen-Shannon Divergence."
        Advances in Neural Information Processing Systems 33 (2020).
    """

    def __init__(self, alpha=None, beta=1.0, dual=True, invert_alpha=True, device=None, **kwargs):
        super(GeometricJSLossTrainablePrior, self).__init__(**kwargs)
    
        if alpha is not None:
            if device is not None:
                self.alpha = torch.nn.Parameter(torch.tensor([alpha]).to(device))
            else:
                self.alpha = torch.nn.Parameter(torch.tensor([alpha]).to(device))

        else:
            if device is not None:
                self.alpha = torch.nn.Parameter(torch.rand(1).to(device))
            else:
                self.alpha = torch.nn.Parameter(torch.rand(1).to(device))
            
        self.mean_prior = torch.nn.Parameter(torch.zeros(10).to(device))
        self.logvar_prior = torch.nn.Parameter(torch.zeros(10).to(device))

        self.beta = beta
        self.dual = dual
        self.invert_alpha = invert_alpha
        self.device = device


    def __call__(self, data, recon_data, latent_dist, is_train, storer,
                 **kwargs):
        storer = self._pre_call(is_train, storer)
        
        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        gjs_loss = _gjs_normal_loss_train_prior(*latent_dist,
                                    mean_prior=self.mean_prior, 
                                    logvar_prior=self.logvar_prior,
                                    dual=self.dual,
                                    a=self.alpha,
                                    invert_alpha=self.invert_alpha,
                                    storer=storer)
        loss = rec_loss + self.beta * gjs_loss

        if storer is not None:
            storer['loss'] += [loss.item()]

        return loss, rec_loss


# HELPERS
def _reconstruction_loss(data, recon_data, distribution="bernoulli",
                         storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data.
    I.e. negative log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines
        the loss Bernoulli corresponds to a binary cross entropy (bse) loss
        and is the most commonly used. It has the issue that it doesn't
        penalize the same way (0.1,0.2) and (0.4,0.5), which might not be
        optimal. Gaussian distribution corresponds to MSE, and is sometimes
        used, but hard to train because it ends up focusing only a few pixels
        that are very wrong. Laplace distribution corresponds to L1 solves
        partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, _, _ = recon_data.size()
    # is_colored = n_chan == 3
    recon_data = torch.clamp(recon_data, 0, 1)
    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        # loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
        loss = F.mse_loss(recon_data, data, reduction="sum")
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything
        # for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3
        # emperical value to give similar values than bernoulli => same HP.
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    # if storer is not None:
    #     storer['recon_loss'] += [loss.item()]

    return loss


def _kl_normal_loss(m_1: torch.Tensor,
                    lv_1: torch.Tensor,
                    m_2: torch.Tensor,
                    lv_2: torch.Tensor,
                    term: Optional[str] = '',
                    storer: Optional[Dict] = None
                    ) -> torch.Tensor:
    """Calculates the KL divergence between two normal distributions
    with diagonal covariance matrices."""
    latent_dim = m_1.size(1)
    latent_kl = (0.5 * (-1 + (lv_2 - lv_1) + lv_1.exp() / lv_2.exp()
                 + (m_2 - m_1).pow(2) / lv_2.exp()).mean(dim=0))
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss' + str(term)] += [total_kl.item()]
        for i in range(latent_dim):
            storer['kl_loss' + str(term) + f'_{i}'] += [latent_kl[i].item()]

    return total_kl


def _get_mu_var(m_1, v_1, m_2, v_2, a=0.5, storer=None):
    """Get mean and standard deviation of geometric mean distribution."""
    v_a = 1 / ((1 - a) / v_1 + a / v_2)
    m_a = v_a * ((1 - a) * m_1 / v_1 + a * m_2 / v_2)

    return m_a, v_a


def _gjs_normal_loss(mean, logvar, dual=False, a=0.5, invert_alpha=True,
                     storer=None, record_alpha_range=False):
    var = logvar.exp()
    mean_0 = torch.zeros_like(mean)
    var_0 = torch.ones_like(var)

    if invert_alpha:
        mean_a, var_a = _get_mu_var(
            mean, var, mean_0, var_0, a=1-a, storer=storer)
    else:
        mean_a, var_a = _get_mu_var(
            mean, var, mean_0, var_0, a=a, storer=storer)

    var_a = torch.log(var_a)
    var_0 = torch.log(var_0)
    var = torch.log(var)

    if dual:
        kl_1 = _kl_normal_loss(
            mean_a, var_a, mean, var, term=1, storer=storer)
        kl_2 = _kl_normal_loss(
            mean_a, var_a, mean_0, var_0, term=2, storer=storer)
    else:
        kl_1 = _kl_normal_loss(
            mean, var, mean_a, var_a, term=1, storer=storer)
        kl_2 = _kl_normal_loss(
            mean_0, var_0, mean_a, var_a, term=2, storer=storer)
    with torch.no_grad():
        _ = _kl_normal_loss(
            mean, var, mean_0, var_0, term='rkl', storer=storer)

    total_gjs = (1 - a) * kl_1 + a * kl_2

    if storer is not None:
        storer_label = 'gjs_loss'
        if dual:
            storer_label += '_dual'
        if invert_alpha:
            storer_label += '_invert'
        storer[storer_label] += [total_gjs.item()]

        # Record what the alpha landscape looks like if record_alpha_range
        if record_alpha_range:
            storer_label = 'gjs_loss'
            if dual:
                storer_label += '_dual'
            if invert_alpha:
                storer_label += '_invert'
            with torch.no_grad():
                for i in range(101):
                    gjs = _gjs_normal_loss(
                        mean, logvar,
                        dual=False, a=i/100, invert_alpha=True, storer=None)
                    storer[f"storer_label_alpha_test={i/100}"] += [gjs.item()]

    return total_gjs


def _mmd_loss(mean: Tensor,
              logvar: Tensor,
              storer: Optional[Dict] = None
              ) -> Tensor:
    """Calculates the maximum mean discrepancy between latent distributions."""
    _, latent_dim = mean.shape
    z = torch.cat((mean, logvar), axis=1)

    true_samples = torch.randn((200, 2 * latent_dim), requires_grad=False)
    true_samples = true_samples.to(z.device)

    x_kernel = _compute_kernel(true_samples, true_samples)
    y_kernel = _compute_kernel(z, z)
    xy_kernel = _compute_kernel(true_samples, z)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    if storer is not None:
        storer['mmd_loss'] += [mmd.item()]

    return mmd


def _compute_kernel(x: Tensor, y: Tensor) -> Tensor:
    """Calculate kernel for maximum mean discrepancy loss."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    kernel_output = torch.exp(-kernel_input)  # (x_size, y_size)

    return kernel_output


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data,
                               is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(
            batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx
