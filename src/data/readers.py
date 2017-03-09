"""
Copyright © Divyanshu Kakwani 2017, all rights reserved.

This module contains several dataset reader classes.

A reader encapsulates:
    The storage format of the data on the disk
    The reading order of the data points in the data
    The per-datapoint preprocessing operations
"""

import os
import sys
import random
import json
import numpy as np

from skimage import io
from skimage.transform import resize
from os.path import basename


def _listdir(dirpath):
    """Works same as os.listdir except that it uses absolute
    addresses instead of relative.
    """
    names = os.listdir(dirpath)
    return [os.path.join(dirpath, n) for n in names]


def _print_progressbar(done, total):
    done = min(total, done)
    per_30 = int((done/total)*30)
    ratio = '%s/%s' % (done, total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-30s] %s" % ('='*per_30, ratio))
    sys.stdout.flush()


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

    def _read_part(self, i, j):
        images = (io.imread(fname) for fname, _ in self.data[i:j])
        X = [resize(im, self.image_shape) for im in images]
        Y = [cat for _, cat in self.data[i:j]]
        return X, Y

    def _read_full(self):
        tot = len(self.data)
        at_once = max(20, tot//20)
        X, Y = [], []
        _print_progressbar(0, tot)
        for i in range(0, len(self.data), at_once):
            X_sub, Y_sub = self._read_part(i, i + at_once)
            X.extend(X_sub)
            Y.extend(Y_sub)
            _print_progressbar(i+at_once, tot)
        return (np.array(X), np.array(Y))

    def _read_in_batches(self):
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            X, Y = self._read_part(start, end)
            yield (np.array(X), np.array(Y))


class ADReader:
    """Annotated Data Reader
    """
    def __init__(self, images_dirpath, labels_dirpath,
                 image_shape=(227, 227, 3), batched=False,
                 batch_size=1024, strategy='random'):
        self.labels_dirpath = labels_dirpath
        self.image_shape = image_shape
        self.batched = batched
        self.batch_size = batch_size if batched else None
        self.strategy = strategy

        if batched is False:
            self.read = self._read_full
        else:
            self.read = self._read_in_batches

        self.labels_subdirpaths = _listdir(self.labels_dirpath)
        self.labels = {}  # maps filenames to a 4-tuple representing the bbox
        for path in self.labels_subdirpaths:
            with open(path, 'r') as fp:
                labels_cat = json.load(fp)
                for l in labels_cat:
                    fname = l['filename']
                    if len(l['annotations']) > 0:
                        fa = l['annotations'][0]    # first annotation
                        coords = [fa['height'], fa['width'], fa['x'], fa['y']]
                        self.labels[fname] = coords

        subdir_paths = [p for p in _listdir(images_dirpath)
                        if os.path.isdir(p)]
        self.img_paths = []
        for path in subdir_paths:
            self.img_paths.extend(img_path for img_path in _listdir(path)
                                  if basename(img_path) in self.labels)
        random.shuffle(self.img_paths)

    def _read_part(self, i, j):
        """
        Reads paths contained in img_paths[i:j] and prepares the data
        in the standard format (X, Y) where X contains the images and
        Y contains the transformed coordinates.
        """
        X, Y = [], []
        for img_path in self.img_paths[i:j]:
            image = io.imread(img_path)
            fname = basename(img_path)
            rsz_factor_x = image.shape[1] / self.image_shape[1]
            rsz_factor_y = image.shape[0] / self.image_shape[0]
            coords = self.labels[basename(fname)]
            transed_coords = [coords[0]/rsz_factor_y, coords[1]/rsz_factor_x,
                              coords[2]/rsz_factor_x, coords[3]/rsz_factor_y]
            X.append(resize(image, self.image_shape))
            Y.append(transed_coords)
        return X, Y

    def _read_full(self):
        tot = len(self.img_paths)
        at_once = max(20, tot//20)
        X, Y = [], []
        _print_progressbar(0, tot)
        for i in range(0, len(self.img_paths), at_once):
            X_sub, Y_sub = self._read_part(i, i + at_once)
            X.extend(X_sub)
            Y.extend(Y_sub)
            _print_progressbar(i+at_once, tot)
        return (np.array(X), np.array(Y))

    def _read_in_batches(self):
        for start in range(0, len(self.img_paths), self.batch_size):
            end = start + self.batch_size
            X, Y = self._read_part(start, end)
            yield np.array(X), np.array(Y)


class ImagesReader:
    """Reads raw images from dirpath
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

        self.img_paths = [fname for fname in _listdir(dirpath)]
        random.shuffle(self.img_paths)

    def _read_part(self, i, j):
        images = (io.imread(img_path) for img_path in self.img_paths[i:j])
        X = np.array([resize(im, self.image_shape) for im in images])
        return X

    def _read_full(self):
        return self._read_part(0, len(self.img_paths))

    def _read_in_batches(self):
        for start in range(0, len(self.img_paths), self.batch_size):
            end = start + self.batch_size
            yield self._read_part(start, end)
