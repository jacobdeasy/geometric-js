"""Training module."""

import logging
import os
import torch
import pdb

from collections import defaultdict
from timeit import default_timer
from tqdm import trange

from vae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"


class IntegrativeTrainer():
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
                 is_progress_bar=True,
                 loss_optimizer=None,
                 batch_size=64,
                 record_alpha_range=False):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(
            os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.logger.info(f"Training Device: {self.device}")
        self.loss_optimizer = loss_optimizer
        self.batch_size = batch_size
        self.record_alpha_range=record_alpha_range

    def __call__(self, data,
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
        self.batches = int(len(data) / self.batch_size)
        for epoch in range(epochs):
            storer = defaultdict(list)
            avg_rec_loss = self._train_epoch(data, storer, epoch)
            storer['recon_loss'] += [avg_rec_loss]
            self.logger.info(
                f'Epoch: {epoch+1} Average reconstruction loss per image: {avg_rec_loss:.2f}')
            if self.loss_optimizer is not None:
                self.losses_logger.log(epoch, storer, 
                                       alpha_parameter=self.loss_f.alpha,
                                       mean = self.loss_f.mean_prior,
                                       logvar = self.loss_f.logvar_prior)
            else:
                self.losses_logger.log(epoch, storer)

            if epoch % checkpoint_every == 0:
                save_model(
                    self.model, self.save_dir, filename=f"model-{epoch}.pt")

        save_model(self.model, self.save_dir, filename=f"model-{epoch+1}.pt")

        self.model.eval()

        dt = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {dt:.1f} min.')

    def _train_epoch(self, data, storer, epoch):
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
        
        # Shuffle the data:
        data = data[torch.randperm(data.size()[0])]
        for i in tqdm(range(self.batches)):
            data_sample = data[i:i + self.batch_size]
            iter_loss, iter_rec_loss = self._train_iteration(data_sample, storer)
            epoch_rec_loss += iter_rec_loss
            t.set_postfix(loss=iter_rec_loss)
            t.update()

        mean_epoch_rec_loss = epoch_rec_loss / len(self.batches)

        return mean_epoch_rec_loss

    def _train_iteration(self, data, storer, clean_data=None):
        """
        ->  Trains the model for one iteration on a batch of data.
        ->  If the loss function passed in the initialisation of the
            Trainer class has trainable parameters then these parameters
            are iterated by this function in addition to the model
            parameters.
        ->  If a noise de-noising experiment is being carried out then
            the reconstruction loss is calculated by comparing the output
            of the VAE decoder with the input image without noise added. In
            this case, a clean batch of images with no noise added is also
            passed via the clean_data parameter.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        clean_data: torch.Tensor
            A batch of data without added noise.
            Shape : (batch_size, channel, height, width).
        """

       
        data = data.to(self.device)

        try:
            if self.loss_optimizer is not None:
                recon_batch, latent_dist, latent_sample = self.model(data)
                loss, rec_loss = self.loss_f(
                    data, recon_batch, latent_dist, self.model.training, storer, record_alpha_range=self.record_alpha_range,
                    latent_sample=latent_sample)
                self.loss_optimizer.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.loss_optimizer.step()
                self.optimizer.step()
                with torch.no_grad():
                    if hasattr(self.loss_f, 'alpha'):
                        self.loss_f.alpha.clamp_(0, 1)
            else:
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

    def log(self, epoch, losses_storer, alpha_parameter=None, mean=None, logvar=None):
        """Write to the log file """
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
