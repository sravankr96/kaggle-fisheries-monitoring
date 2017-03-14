"""
Copyright © Sr@1 2017, All rights reserved

    *The NN model is imported from /model_archs/raw_image_classifier.py

    *The Hyper parameters can be set here

    *The data is split into train and test sets for training and validation

    *The run_classifier() modul is used to:
        *Train the model and validate it
        *save the best weights of the model
        *save the model to JSON file
"""


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam
from model_archs.raw_image_classifier import RawImageClassifier
from data.readers import CDReader
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

input_shape = (90, 160, 3)

hyper_params = {'optimizer': adam(lr=1e-3),
                'objective': 'categorical_crossentropy',
                'input_shape': input_shape,
                'activation': 'relu'}
epochs = 10

# Reading the datasets using categorical dataset reader
print("Reading dataset for Classification")
reader = CDReader(dirpath='./../datasets/train', image_shape=input_shape)
X_all, Y_all = reader.read()


# Encoding Labels
Y_all = LabelEncoder().fit_transform(Y_all)
Y_all = np_utils.to_categorical(Y_all)

print("\n", X_all.shape, "\n", Y_all.shape)


# Loading the model from raw-image-classifier.py
model = RawImageClassifier(hyper_params)


def run_classifier():

    print("Starting Classifier")
    # Early stopping and fitting the data into the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

    # Creating checkpoints using callbacks
    filepath = "./../outputs/classifier-weights-best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

    model.fit(X_all, Y_all, batch_size=64, nb_epoch=epochs,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping, checkpoint])

    # Saving the model to a json file
    f1 = open('./../outputs/raw-image-classifier-model.json', 'w')
    f1.write(model.to_json())
    f1.close()
