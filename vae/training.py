"""Training module."""

import logging
import os
import torch

from collections import defaultdict
from timeit import default_timer
from tqdm import trange

from vae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(
            os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info(f"Training Device: {self.device}")

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            avg_rec_loss = self._train_epoch(data_loader, storer, epoch)
            storer['recon_loss'] += [avg_rec_loss]
            self.logger.info(
                f'Epoch: {epoch+1} Average reconstruction loss per image: {avg_rec_loss:.2f}')
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

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_rec_loss = 0.
        kwargs = dict(desc=f"Epoch {epoch+1}", leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, _) in enumerate(data_loader):
                iter_loss, iter_rec_loss = self._train_iteration(data, storer)
                epoch_rec_loss += iter_rec_loss

                t.set_postfix(loss=iter_rec_loss)
                t.update()

        mean_epoch_rec_loss = epoch_rec_loss / len(data_loader)

        return mean_epoch_rec_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss, rec_loss = self.loss_f(
                data, recon_batch, latent_dist, self.model.training, storer,
                latent_sample=latent_sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(
                data, self.model, self.optimizer, storer)

        return loss.item(), rec_loss.item()


class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(
                str(item) for item in [epoch, k, sum(v) / len(v)])
            self.logger.debug(log_string)
