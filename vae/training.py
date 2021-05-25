"""Training module."""

import logging
import os
import torch

from collections import defaultdict
from logging import Logger
from timeit import default_timer
from tqdm import trange
from typing import Any, Dict, Optional, Tuple

from vae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():

    def __init__(self,
                 model: Any,
                 optimizer: torch.optim.Optimizer,
                 loss_f: Any,
                 device: Optional[torch.device] = torch.device("cpu"),
                 logger: Optional[Logger] = logging.getLogger(__name__),
                 save_dir: Optional[str] = "results",
                 gif_visualizer: Optional[Any] = None,
                 is_progress_bar: Optional[bool] = True,
                 loss_optimizer: Optional[bool] = None,
                 denoise: Optional[bool] = False,
                 record_alpha_range: Optional[bool] = False
                 ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.gif_visualizer = gif_visualizer
        self.is_progress_bar = is_progress_bar
        self.loss_optimizer = loss_optimizer
        self.denoise = denoise
        self.record_alpha_range = record_alpha_range

        self.losses_logger = LossesLogger(
            os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.logger.info(f"Training Device: {self.device}")

    def __call__(self,
                 data_loader: torch.utils.data.DataLoader,
                 epochs: Optional[int] = 10,
                 checkpoint_every: Optional[int] = 10
                 ) -> None:
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            avg_rec_loss = self._train_epoch(data_loader, storer, epoch)
            storer['recon_loss'] += [avg_rec_loss]
            self.logger.info(
                f'Epoch: {epoch+1} Avg recon loss / image: {avg_rec_loss:.2f}')
            if self.loss_optimizer is not None:
                self.losses_logger.log(epoch,
                                       storer,
                                       alpha_parameter=self.loss_f.alpha)
            else:
                self.losses_logger.log(epoch, storer)

            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(
                    self.model, self.save_dir, filename=f"model-{epoch}.pt")

        save_model(self.model, self.save_dir, filename=f"model-{epoch+1}.pt")

        if self.gif_visualizer is not None:
            self.gif_visualizer.save_reset()

        self.model.eval()

        dt = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {dt:.1f} min.')

    def _train_epoch(self,
                     data_loader: torch.utils.data.DataLoader,
                     storer: Dict,
                     epoch: int
                     ) -> float:
        epoch_rec_loss = 0.
        kwargs = dict(desc=f"Epoch {epoch+1}", leave=False,
                      disable=not self.is_progress_bar)

        with trange(len(data_loader), **kwargs) as t:
            if self.denoise:
                # If denoising experiment, calculate the reconstruction error
                # by comparing the output of the vae decoder with clean
                for _, (noisy_data, clean_data) in enumerate(data_loader):
                    iter_loss, iter_rec_loss = self._train_iteration(
                        data=noisy_data, storer=storer, clean_data=clean_data)
                    epoch_rec_loss += iter_rec_loss
                    t.set_postfix(loss=iter_rec_loss)
                    t.update()
            else:
                for _, (data, _) in enumerate(data_loader):
                    iter_loss, iter_rec_loss = self._train_iteration(
                        data, storer)
                    epoch_rec_loss += iter_rec_loss
                    t.set_postfix(loss=iter_rec_loss)
                    t.update()

        return epoch_rec_loss / len(data_loader)

    def _train_iteration(self,
                         data: torch.Tensor,
                         storer: Dict,
                         clean_data: Optional[torch.Tensor] = None
                         ) -> Tuple[float, float]:
        batch_size, channel, height, width = data.size()
        data_in = data.to(self.device)
        if clean_data is not None:
            data_out = clean_data.to(self.device)
        else:
            data_out = data.to(self.device)

        try:
            # Iterate loss parameters if a loss optimiser is passed:
            recon_batch, latent_dist, latent_sample = self.model(data_in)
            loss, rec_loss = self.loss_f(
                data_out, recon_batch, latent_dist, self.model.training,
                storer, record_alpha_range=self.record_alpha_range,
                latent_sample=latent_sample)
            if self.loss_optimizer is not None:
                self.loss_optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.loss_optimizer is not None:
                self.loss_optimizer.step()
                with torch.no_grad():
                    if hasattr(self.loss_f, 'alpha'):
                        self.loss_f.alpha.clamp_(0, 1)

        except ValueError:
            # For losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(
                data, self.model, self.optimizer, storer)

        return loss.item(), rec_loss.item()


class LossesLogger(object):

    def __init__(self, file_path_name):
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)
        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer,
            alpha_parameter=None, mean=None, logvar=None):
        for k, v in losses_storer.items():
            log_string = ",".join(
                str(item) for item in [epoch, k, sum(v) / len(v)])
            self.logger.debug(log_string)
        if alpha_parameter is not None:
            self.logger.debug(f"{epoch},alpha,{alpha_parameter.item()}")
        if mean is not None:
            for i in range(len(mean)):
                self.logger.debug(f"{epoch},mean_{i+1},{mean[i].item()}")
        if logvar is not None:
            var = logvar.exp()
            for i in range(len(var)):
                self.logger.debug(f"{epoch},var_{i+1},{var[i].item()}")
