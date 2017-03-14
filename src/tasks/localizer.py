"""
Copyright © Sr@1 2017, All rights reserved

    *The NN model is imported from /model_archs/localizer_as_regr.py

    *The Hyper parameters can be set here

    *The data is split into train and test sets for training and validation

    *The run_localizer() modul is used to:
        *Train the model and validate it
        *save the best weights of the model
        *save the model to JSON file
"""


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam
from model_archs.localizer_as_regr import LocalizerAsRegrModel
from data.readers import ADReader


input_shape = (90, 130, 3)
epochs = 50

hyper_params = {'optimizer': adam(lr=1e-4),
                'objective': 'mse',
                'input_shape': input_shape,
                'activation': 'relu'}

print("Reading dataset for localisation")
reader = ADReader(images_dirpath='./../datasets/train',
                  labels_dirpath='./../datasets/labels', image_shape=input_shape)
X_all, Y_all = reader.read()

# Loading the model architecture
model = LocalizerAsRegrModel(hyper_params)


def run_localizer():

    print("Starting Loacaliser")
    # Early stopping and fitting the data into the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

    # Creating checkpoints using callbacks
    filepath = "./../outputs/localizer-as-regr-model-weights-best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

    model.fit(X_all, Y_all, batch_size=64, nb_epoch=epochs,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping, checkpoint])

    # Saving the model to a json file
    f1 = open('./../outputs/localizer-as-regr-model.json', 'w')
    f1.write(model.to_json())
    f1.close()
