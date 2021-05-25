"""Main training module."""

import argparse
import logging
import os
import sys
import torch.optim as optim

from typing import List

from vae import Evaluator, Trainer
from vae.models.losses import LOSSES, RECON_DIST, get_loss_f
from vae.models.vae import VAE
from vae.utils.modelIO import save_model, load_model, load_metadata
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed,
                           get_n_param)
from utils.visualize import GifTraversalsTraining


RES_DIR = 'results_new'


def parse_arguments(args_to_parse: List):
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
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading.")
    general.add_argument('--no-progress-bar',
                         action='store_true', default=False,
                         help='Disables progress bar.')
    general.add_argument('--no-cuda',
                         action='store_true', default=False,
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed',
                         type=int, default=1234,
                         help='Random seed. `None` for stochastic behaviour.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=10,
                          help='Save a the trained model every n epochs.')
    training.add_argument('-d', '--dataset',
                          default='dsprites', choices=DATASETS,
                          help="Path to training data.")
    training.add_argument('-e', '--epochs',
                          type=int, default=30,
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size',
                          type=int, default=64,
                          help='Batch size for training.')
    training.add_argument('--lr', '--learning-rate',
                          type=float, default=1e-4,
                          help='Learning rate.')
    training.add_argument('--noise',
                          type=float, default=None,
                          help='Added noise to input images.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-z', '--latent-dim',
                       type=int, default=10,
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default='GJS', choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist',
                       default='bernoulli', choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal',
                       type=float, default=10000,
                       help="Number of annealing steps for regularisation.")

    # Loss Specific Options
    GJS = parser.add_argument_group('Geometric Jensen-Shannon parameters')
    GJS.add_argument('--GJS-A',
                     type=float, default=0.5,
                     help='Skew of geometric-JS (alpha in the paper).')
    GJS.add_argument('--GJS-B',
                     type=float, default=1.0,
                     help='Weight of the skew geometric-JS.')
    GJS.add_argument('--GJS-invA',
                     type=bool, default=True,
                     help='Whether to invert alpha.')

    betaH = parser.add_argument_group('BetaH parameters')
    betaH.add_argument('--betaH-B',
                       type=float, default=4.0,
                       help='Weight of the KL (beta in the paper).')

    MMD = parser.add_argument_group('MMD parameters')
    MMD.add_argument('--MMD-B',
                     type=float, default=500.0,
                     help='Weight of the MMD (lambda in the paper).')

    betaB = parser.add_argument_group('BetaB parameters')
    betaB.add_argument('--betaB-initC',
                       type=float, default=0.0,
                       help='Starting annealed capacity.')
    betaB.add_argument('--betaB-finC',
                       type=float, default=25.0,
                       help='Final annealed capacity.')
    betaB.add_argument('--betaB-G',
                       type=float, default=100,
                       help='Weight of the KL (gamma in the paper).')

    factor = parser.add_argument_group('factor VAE parameters')
    factor.add_argument('--factor-G',
                        type=float, default=6.0,
                        help='Weight of the TC term (gamma in the paper).')
    factor.add_argument('--lr-disc',
                        type=float, default=5e-5,
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae parameters')
    btcvae.add_argument('--btcvae-A',
                        type=float, default=1.0,
                        help='Weight of the MI term (alpha in the paper).')
    btcvae.add_argument('--btcvae-G',
                        type=float, default=1.0,
                        help='Weight of the dim-wise KL (gamma in the paper).')
    btcvae.add_argument('--btcvae-B',
                        type=float, default=6.0,
                        help='Weight of the TC term (beta in the paper).')

    # Learning options
    evaluation = parser.add_argument_group('Evaluation options')
    evaluation.add_argument('--is-eval-only',
                            action='store_true', default=False,
                            help='Whether to only evaluate model `name`.')
    evaluation.add_argument('--is-metrics',
                            action='store_true', default=False,
                            help='Whether to compute disentanglement metrics.')

    args = parser.parse_args(args_to_parse)
    print(args)

    return args


def main(args: argparse.Namespace):
    """Main train and evaluation function."""
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s - %(funcName)s: %(message)s', "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    stream = logging.StreamHandler()
    stream.setLevel("INFO")
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info(
        f"Root directory for saving and loading experiments: {exp_dir}")

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)

        if args.loss == "factor":
            logger.info(
                "FactorVae needs 2 batches per iteration." +
                "To replicate this behavior, double batch size and epochs.")
            args.batch_size *= 2
            args.epochs *= 2

        # PREPARES DATA
        train_loader = get_dataloaders(args.dataset,
                                       noise=args.noise,
                                       batch_size=args.batch_size,
                                       logger=logger)
        logger.info(
            f"Train {args.dataset} with {len(train_loader.dataset)} samples")

        # PREPARES MODEL
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        model = VAE(args.img_size, args.latent_dim)
        logger.info(f'Num parameters in model: {get_n_param(model)}')

        # TRAINS
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model = model.to(device)
        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))

        if args.loss in ['tdGJS', 'tGJS']:
            loss_optimizer = optim.Adam(loss_f.parameters(), lr=args.lr)
        else:
            loss_optimizer = None
        print(loss_optimizer)
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          gif_visualizer=gif_visualizer,
                          loss_optimizer=loss_optimizer,
                          denoise=args.noise is not None)
        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    # Eval
    model = load_model(exp_dir, is_gpu=not args.no_cuda)
    metadata = load_metadata(exp_dir)

    test_loader = get_dataloaders(metadata["dataset"],
                                  noise=args.noise,
                                  train=False,
                                  batch_size=128,
                                  logger=logger)
    loss_f = get_loss_f(args.loss,
                        n_data=len(test_loader.dataset),
                        device=device,
                        **vars(args))
    evaluator = Evaluator(model, loss_f,
                          device=device,
                          is_metrics=args.is_metrics,
                          is_train=False,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          denoise=args.noise is not None)
    evaluator(test_loader)

    # Train set also
    test_loader = get_dataloaders(metadata["dataset"],
                                  train=True,
                                  batch_size=128,
                                  logger=logger)
    loss_f = get_loss_f(args.loss,
                        n_data=len(test_loader.dataset),
                        device=device,
                        **vars(args))
    evaluator = Evaluator(model, loss_f,
                          device=device,
                          is_metrics=args.is_metrics,
                          is_train=True,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar)
    evaluator(test_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
