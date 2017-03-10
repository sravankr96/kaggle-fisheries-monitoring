"""
Copyright © Sr@1 2017, All rights reserved.

    *The module contains the architecture of a localization network

    *Network is built analogous to VGG network

    *The Hyper parameters of the network are passed from the /tasks/localizer.py module


"""

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, Dropout, Dense, InputLayer
import keras.backend as K


class LocalizerAsRegrModel(Sequential):

    def __init__(self, hyper_params):
        self.optimizer = hyper_params['optimizer']
        self.objective = hyper_params['objective']
        self.shape = hyper_params['input_shape']
        self.activation = hyper_params['activation']

        # Building the model
        def center_normalize(x):
            return (x - K.mean(x)) / K.std(x)

        super(LocalizerAsRegrModel, self).__init__()

        self.add(Activation(activation=center_normalize, input_shape=self.shape))

        # 256
        self.add(Convolution2D(16, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        # 128
        self.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 64
        self.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, subsample=(2, 2),
                               dim_ordering='tf'))
        self.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 32
        self.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, subsample=(2, 2),
                               dim_ordering='tf'))
        self.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 16
        self.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, subsample=(2, 2),
                               dim_ordering='tf'))
        self.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 8
        self.add(MaxPooling2D(pool_size=(8, 8), dim_ordering='tf'))
        self.add(Dropout(0.5))

        self.add(Dense(4))

        self.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
