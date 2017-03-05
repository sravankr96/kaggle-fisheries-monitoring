"""
Copyright © Sr@1 2017, All rights reserved.

    *The module contains the architecture of a localization network

    *Network is built analogous to VGG network

    *The Hyper prameters of the network are passed from the /tasks/localizer.py module


"""

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, Dropout, Dense
import keras.backend as K


class LocalizerAsRegrModel:

    def __init__(self, hyper_params):
        self.optimizer = hyper_params['optimizer']
        self.objective = hyper_params['objective']
        self.input_shape = hyper_params['input_shape']
        self.activation = hyper_params['activation']

        # Building the model
        def center_normalize(x):
            return (x - K.mean(x)) / K.std(x)

        self.model = Sequential()

        self.model.add(Activation(activation=center_normalize, input_shape=self.input_shape))

        # 256
        self.model.add(Convolution2D(16, 7, 7, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

        # 128
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 64
        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, strides=(2, 2),
                                     dim_ordering='tf'))
        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 32
        self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, strides=(2, 2),
                                     dim_ordering='tf'))
        self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 16
        self.model.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, strides=(2, 2),
                                     dim_ordering='tf'))
        self.model.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))
        self.model.add(Convolution2D(256, 3, 3, border_mode='same', activation=self.activation, dim_ordering='tf'))

        # 8
        self.model.add(MaxPooling2D(pool_size=(8, 8), dim_ordering='tf'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(4))

    def _get_model(self):

        self.model.compile(loss=self.objective, optimizer=self.optimizer, metrics=['accuracy'])
        return self.model

    def fit(self, x, y, batch_size, nb_epoch=1,
            validation_split=0.2, verbose=1, shuffle=True, callbacks=[]):

        return self.model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,
                              validation_split=validation_split, verbose=verbose, shuffle=shuffle, callbacks=callbacks)

    def predict(self, x, verbose=1):

        return self.model.predict(x, verbose=verbose)

    def to_json(self):

        return self.model.to_json()
