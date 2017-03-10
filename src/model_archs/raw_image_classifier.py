"""
Copyright © Sr@1 2017, All rights reserved.

    *The module contains the architecture of a raw image calssifier network

    *Network is built analogous to VGG network

    *The Hyper parameters of the network are passed from the /tasks/classifier.py module


"""

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten
import keras.backend as K


class RawImageClassifier(Sequential):

    def __init__(self, hyper_params):
        self.optimizer = hyper_params['optimizer']
        self.objective = hyper_params['objective']
        self.shape = hyper_params['input_shape']
        self.activation = hyper_params['activation']

        # Building the model
        def center_normalize(x):
            return (x - K.mean(x)) / K.std(x)

        super(RawImageClassifier, self).__init__()

        self.add(Activation(activation=center_normalize, input_shape=self.shape))

        self.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        self.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        self.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        self.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        self.add(Flatten())
        self.add(Dense(256, activation=self.activation))
        self.add(Dropout(0.5))

        self.add(Dense(64, activation=self.activation))
        self.add(Dropout(0.5))

        # Number of Fish classes = 8
        self.add(Dense(8))
        self.add(Activation('softmax'))

        self.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
