import glob
import os
import torch
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import numpy as np
import cv2


class DogsCatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: directory to images
        :param transform: Optional transform to be applied to images
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        images_list = os.listdir(self.root_dir)
        image_name = os.path.join(self.root_dir, images_list[idx])
        image = mpimg.imread(image_name)

        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        category_list = images_list[idx].split(".")
        category = category_list[0]

        if category == "cat":
            category = 0
        else:
            category = 1

        sample = {'image': image, 'category': category}

        if self.transform:
            sample = self.transform(sample)

        return sample

# transforms


class Normalize(object):
    """Convert a color image to grayscale and normalize color range to [0,1]."""

    def __call__(self, sample):
        image, category = sample['image'], sample['category']
        image_copy = np.copy(image)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy/255

        return {'image': image_copy, 'category': category}


class Rescale(object):
    """Rescale the image in a sample to a given size"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, category = sample['image'], sample['category']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        return {'image': img, 'category': category}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, category = sample['image'], sample['category']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'category': category}


class ToTensor(object):
    """Convert ndarrays to Tensors"""
    def __call__(self, sample):
        image, category = sample['image'], sample['category']

        # if image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image), 'category': category}