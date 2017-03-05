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
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from model_archs.localizer_as_regr import LocalizerAsRegrModel
from data.readers import ADReader

hyper_params = {'optimizer': adam(lr=1e-4),
                'objective': 'mse',
                'input_shape': (256, 256, 3),
                'activation': 'relu'}
epochs = 10
reader = ADReader(images_dirpath='./../dataset/train', labels_dirpath='./../dataset/labels')
X_all, Y_all = reader._read_full()

# Creating stratified test and validation split
X_train, X_valid, Y_train, Y_valid = train_test_split(X_all, Y_all,
                                                      test_size=0.2, random_state=23,
                                                      stratify=Y_all)
# Loading the model architecture
model = LocalizerAsRegrModel(hyper_params)


def run_localizer():

    # Early stopping and fitting the data into the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

    # Creating checkpoints using callbacks
    filepath = "weights-best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

    model.fit(X_train, Y_train, batch_size=64, nb_epoch=epochs,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping, checkpoint])

    # Validating the model
    preds = model.predict(X_valid, verbose=1)
    print("Validation Log Loss: {}".format(log_loss(Y_valid, preds)))

    # Saving the model to a json file
    f1 = open('model.json', 'w')
    f1.write(model.to_json())
