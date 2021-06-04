"""Dataset loading module.

Adapted from: https://github.com/YannDubs/disentangling-vae"""


import abc
import glob
import hashlib
import h5py
import logging
import numpy as np
import os
import subprocess
import tarfile
import torch
import urllib.request
import zipfile

from PIL import Image
from skimage.io import imread
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from typing import Any, List, Optional, Tuple


DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist":    "MNIST",
                 "fashion":  "FashionMNIST",
                 "nmnist":   "NoisyMNIST",
                 "bmnist":   "BinarizedMNIST",
                 "dsprites": "DSprites",
                 "celeba":   "CelebA",
                 "chairs":   "Chairs"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset: str) -> Dataset:
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError(f"Unkown dataset: {dataset}")


def get_img_size(dataset: str) -> Tuple:
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset: str,
                    train: Optional[bool] = True,
                    noise: Optional[float] = None,
                    root: Optional[str] = None,
                    pin_memory: Optional[bool] = True,
                    batch_size: Optional[int] = 128,
                    logger: Optional[Any] = logging.getLogger(__name__),
                    **kwargs: Any
                    ) -> DataLoader:
    """A generic data loader
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load
    """

    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU
    Dataset = get_dataset(dataset)

    # Initialise the dataset class:
    if root is None:
        if noise == 0.0 or noise is None:
            dataset = Dataset(train=train, logger=logger)
        else:
            dataset = Dataset(train=train, noise=noise, logger=logger)
    else:
        if noise == 0.0 or noise is None:
            dataset = Dataset(train=train, root=root, logger=logger)
        else:
            dataset = Dataset(train=train, noise=noise, root=root,
                              logger=logger)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      pin_memory=pin_memory,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets."""

    def __init__(self,
                 root: str,
                 transforms_list: Optional[List[Any]] = [],
                 logger: Optional[Any] = logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info(f"Downloading {str(type(self))} ...")
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self) -> int:
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def download(self):
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset. Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    background_color = COLOUR_BLACK
    lat_values = {
        'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                          0.16129032, 0.19354839, 0.22580645, 0.25806452,
                          0.29032258, 0.32258065, 0.35483871, 0.38709677,
                          0.41935484, 0.4516129, 0.48387097, 0.51612903,
                          0.5483871, 0.58064516, 0.61290323, 0.64516129,
                          0.67741935, 0.70967742, 0.74193548, 0.77419355,
                          0.80645161, 0.83870968, 0.87096774, 0.90322581,
                          0.93548387, 0.96774194, 1.]),
        'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                          0.16129032, 0.19354839, 0.22580645, 0.25806452,
                          0.29032258, 0.32258065, 0.35483871, 0.38709677,
                          0.41935484, 0.4516129, 0.48387097, 0.51612903,
                          0.5483871, 0.58064516, 0.61290323, 0.64516129,
                          0.67741935, 0.70967742, 0.74193548, 0.77419355,
                          0.80645161, 0.83870968, 0.87096774, 0.90322581,
                          0.93548387, 0.96774194, 1.]),
        'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
        'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                 0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                 1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                 1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                 2.57771705, 2.73882436, 2.89993168, 3.061039,
                                 3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                 3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                 4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                 5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                 5.79986336, 5.96097068, 6.12207799, 6.28318531]),
        'shape': np.array([1., 2., 3.]),
        'color': np.array([1.])}

    def __init__(self,
                 train: Optional[bool] = True,
                 root: Optional[str] = os.path.join(DIR, '../data/dsprites/'),
                 **kwargs: Optional[Any]):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]

        return sample, lat_value


class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self,
                 train: Optional[bool] = True,
                 root: Optional[str] = os.path.join(DIR, '../data/celeba'),
                 **kwargs: Optional[Any]):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (Dataloaders can't return None)
        return img, 0


class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self,
                 train: Optional[bool] = True,
                 root: Optional[str] = os.path.join(DIR, '../data/chairs'),
                 logger: Optional[Any] = logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, train=True, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Pad(2),
                             transforms.ToTensor()
                         ]))


class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, train=True,
                 root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Pad(2),
                             transforms.ToTensor()
                         ]))


class NoisyMNIST(Dataset):
    """Noisy MNIST wrapper."""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, train=True, noise=None,
                 root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__()
        if train:
            mnist_data = torch.load(
                os.path.join(root, 'MNIST', 'processed', 'training.pt'))
        else:
            mnist_data = torch.load(
                os.path.join(root, 'MNIST', 'processed', 'test.pt'))
        self.x = mnist_data[0]

        self.mnist_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor()
        ])
        if noise is not None:
            self.add_noise = AddGaussianNoise(mean=0.0, std=noise)
        self.noise = noise
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input = self.mnist_transforms(self.x[idx:idx + 1])
        if self.noise is not None:
            input = self.add_noise(input)
        output = self.mnist_transforms(self.x[idx:idx + 1])

        return input, output


class BinarizedMNIST(Dataset):
    """ Binarized MNIST dataset, proposed in
    http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf """
    train_file = 'binarized_mnist_train.amat'
    val_file = 'binarized_mnist_valid.amat'
    test_file = 'binarized_mnist_test.amat'
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, train=True, root=os.path.join(DIR, '../data/bmnist'),
                 logger=logging.getLogger(__name__)):
        # we ignore transform.
        self.root = root
        self.train = train  # training set or test set

        if not self._check_exists():
            self.download()

        self.data = self._get_data(train=train)
        self.mnist_transforms = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img = self.mnist_transforms(img)
        # img = transforms.Pad(2)(transforms.ToTensor()(img)).type(torch.FloatTensor)
        return img.float(), torch.tensor(-1)  # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        with h5py.File(os.path.join(self.root, 'data.h5'), 'r') as hf:
            data = hf.get('train' if train else 'test')
            data = np.array(data)
        return data

    def get_mean_img(self):
        return self.data.mean(0).flatten()

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading MNIST with fixed binarization...')
        for dataset in ['train', 'valid', 'test']:
            filename = 'binarized_mnist_{}.amat'.format(dataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(dataset)
            print('Downloading from {}...'.format(url))
            local_filename = os.path.join(self.root, filename)
            urllib.request.urlretrieve(url, local_filename)
            print('Saved to {}'.format(local_filename))

        def filename_to_np(filename):
            with open(filename) as f:
                lines = f.readlines()
            return np.array([[int(i)for i in line.split()] for line in lines]).astype('int8')

        train_data = np.concatenate([filename_to_np(os.path.join(self.root, self.train_file)),
                                     filename_to_np(os.path.join(self.root, self.val_file))])
        test_data = filename_to_np(os.path.join(self.root, self.val_file))
        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('train', data=train_data.reshape(-1, 28, 28))
            hf.create_dataset('test', data=test_data.reshape(-1, 28, 28))
        print('Done!')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data.h5'))


# HELPERS
def preprocess(root: str,
               size: Optional[Tuple[int, int]] = (64, 64),
               img_format: Optional[str] = 'JPEG',
               center_crop: Optional[Tuple[int, int]] = None
               ) -> None:
    """Preprocess a folder of images.

    Parameters
    ----------
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


class AddGaussianNoise(object):
    def __init__(self,
                 mean: Optional[float] = 0.0,
                 std: Optional[float] = 1.0
                 ) -> None:
        self.std = std
        self.mean = mean

    def __call__(self, tensor: Tensor) -> Tensor:
        # return tensor + torch.randn(tensor.size()) * self.std + self.mean
        # Clamp output so image with noise is still greyscale:
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
