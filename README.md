# Constraining Variational Inference with Geometric Jensen-Shannon Divergence [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jacobdeasy/geometric-JS/blob/master/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

#### Abstract
We examine the problem of _controlling divergences_ for latent space regularisation in variational autoencoders. Specifically, when aiming to reconstruct example <img src="https://latex.codecogs.com/gif.latex?x\in\mathbb{R}^{m}" title="x\in\mathbb{R}^{m}" /> via latent space <img src="https://latex.codecogs.com/gif.latex?z\in\mathbb{R}^{n}&space;(n\leq&space;m)" title="z\in\mathbb{R}^{n} (n\leq m)" />, while balancing this against the need for generalisable latent representations. We present a regularisation mechanism based on the skew geometric-Jensen-Shannon divergence <img src="https://latex.codecogs.com/gif.latex?\left(\textrm{JS}^{\textrm{G}_{\alpha}}\right)" title="\left(\textrm{JS}^{\textrm{G}_{\alpha}}\right)" />. We find a variation in <img src="https://latex.codecogs.com/gif.latex?\textrm{JS}^{\textrm{G}_{\alpha}}" title="\textrm{JS}^{\textrm{G}_{\alpha}}" />, motivated by limiting cases, which leads to an intuitive interpolation between forward and reverse KL in the space of both distributions and divergences. We motivate its potential benefits for VAEs through low-dimensional examples, before presenting quantitative and qualitative results. Our experiments demonstrate that skewing our variant of <img src="https://latex.codecogs.com/gif.latex?\textrm{JS}^{\textrm{G}_{\alpha}}" title="\textrm{JS}^{\textrm{G}_{\alpha}}" />, in the context of <img src="https://latex.codecogs.com/gif.latex?\textrm{JS}^{\textrm{G}_{\alpha}}" title="\textrm{JS}^{\textrm{G}_{\alpha}}" />-VAEs, leads to better reconstruction and generation when compared to several baseline VAEs. Our approach is entirely unsupervised and utilises only one hyperparameter which can be easily interpreted in latent space.

**Paper** https://arxiv.org/abs/2006.10599

### Basic requirments
The main package requirments are given in `requirements.txt`

### Running experiments

To generate any of the models given in the paper, use the command:
`python .\main.py <experiment_name> --dataset <dataset> --epochs <n_epochs> --latent-dim <n_latents> --rec-dist <reconstruction_loss> --loss <divergence_loss>`
and add any loss-specific flags (e.g. `--GJS-A` sets the alpha value in skew-geometric Jensen-Shannon divergence). All flags and options can be found in `main.py`.

### Examples 

The `notebooks` folder consists of several notebooks that recreate the experiments presented in the paper. However, the folder relies on a folder `/results` that we have omitted due to size limitations, so each result from Figure 3 onwards relies on an experimental run.

In particular:
- `gjs.ipynb` and `dgjs.ipynb` - showcase the usage of GJS-VAEs and GJS*-VAEs on the four datasets, respectively (Figure 3 in the main text and Appendix B)
- `reconstruction.ipynb` and `reconstruction_grid.ipynb` - output the reconstruction images of each experiment
- `alpha_change.ipynb` - showcases experiments that investigate the influence of \alpha 
- `gaussian_example.ipynb` - provides the examples given in Figure1 in the main text.
- `comparison.py` - provides the examples given in Figure2 in the main text.
