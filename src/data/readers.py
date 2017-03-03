"""
Copyright © Divyanshu Kakwani 201t7, all rights reserved.

This module contains several dataset reader classes.

Each reader reads a specific type of dataset and implements
multiple reading strategies.

"""

import os
import random
import numpy as np

from skimage import io
from skimage.transform import resize


def _listdir(dirpath):
    """Works same as os.listdir except that it uses absolute
    addresses instead of relative.
    """
    names = os.listdir(dirpath)
    return [os.path.join(dirpath, n) for n in names]


class CDReader:
    """Categorical Dataset Reader

    Parameter
    ---------
    dirpath: str
        The absolute path of the directory that contains the dataset
    image_shape: 3-tuple
    batched: Boolean
        Read in batches or not
    batch_size: int
    strategy: str
        specifies the strategy to be used to determine reading order
    """
    def __init__(self, dirpath, image_shape=(227, 227, 3), batched=False,
                 batch_size=1024, strategy='random'):
        if not os.path.isdir(dirpath):
            raise FileNotFoundError()

        self.dirpath = dirpath
        self.image_shape = image_shape
        self.batched = batched
        self.batch_size = batch_size if batched else None
        self.strategy = strategy

        if batched is False:
            self.read = self._read_full
        else:
            self.read = self._read_in_batches

        subdir_paths = [p for p in _listdir(dirpath) if os.path.isdir(p)]
        self.data = []
        for path in subdir_paths:
            cat = os.path.basename(path)
            self.data.extend((fname, cat) for fname in _listdir(path))
        random.shuffle(self.data)

    def _read_full(self):
        images = (io.imread(fname) for fname, _ in self.data)
        X = np.array([resize(im, self.image_shape) for im in images])
        Y = np.array([cat for _, cat in self.data])
        return X, Y

    def _read_in_batches(self):
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            images = (io.imread(fname) for fname, _ in self.data[start:end])
            X = np.array([resize(im, self.image_shape) for im in images])
            Y = np.array([cat for _, cat in self.data[start:end]])
            yield X, Y
