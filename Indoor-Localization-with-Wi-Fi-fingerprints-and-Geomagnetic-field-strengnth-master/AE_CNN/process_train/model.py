import pandas as pd
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import keras.backend as backend
from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Conv2D, Dense, Flatten, Dropout, Input, Reshape, Activation, BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import metrics
from process import data_process
# from keras.callbacks import ReduceLROnPlateau
# from keras import optimizers, losses
# from keras.optimizers import Adam, Adadelta
# from sklearn.model_selection import StratifiedKFold

hidden_nodes = 16 * 16
size = int(np.sqrt(hidden_nodes))

l1 = regularizers.l1(0.0)
l2 = regularizers.l2(0.01)
regilization = l1
VERBOSE = 2  # 0 for turning off logging
# ------------------------------------------------------------------------
# stacked auto encoder (sae)
# ------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
# ------------------------------------------------------------------------
# classifier
# ------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
CLASSIFIER_LOSS = 'categorical_crossentropy'


def param():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 50",
        default=50,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=50,
        type=int)
    parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.90",
        default=0.9,
        type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        # default='256, 128,' + str(hidden_nodes) + ',128, 256',
        default='300,' + str(hidden_nodes) + ',300',
        type=str)
    parser.add_argument(
        "-C",
        "--classifier_hidden_layers",
        help=
        "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default=0.4,
        type=float)
    # parser.add_argument(
    #     "--building_weight",
    #     help=
    #     "weight for building classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    # parser.add_argument(
    #     "--floor_weight",
    #     help=
    #     "weight for floor classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help="number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
        type=float)
    args = parser.parse_args()
    return args


def build_model(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS, batch_size, epochs,
                VERBOSE, RSS_train):
    # create a model based on stacked autoencoder (SAE)
    model = Sequential()
    model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    for units in sae_hidden_layers[1:]:
        model.add(Dense(units, use_bias=SAE_BIAS, activation=SAE_ACTIVATION, activity_regularizer=regilization, ))
        # model.add(BatchNormalization())
        # model.add(Activation(SAE_ACTIVATION))
    model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, ))
    model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS, metrics=['acc'])

    # train the model
    model.fit(RSS_train, RSS_train, batch_size=batch_size, epochs=50, verbose=VERBOSE, shuffle=True,
              validation_data=(RSS_val, RSS_val))
    # remove the decoder part
    num_to_remove = (len(sae_hidden_layers) + 1) // 2
    for i in range(num_to_remove):
        model.pop()
    model.add(Dropout(dropout))
    return model


def build_CNN(output_dim):
    AE_output = Input(batch_shape=(batch_size, size, size, 1))
    x = AE_output
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(output_dim)(x)
    x = BatchNormalization()(x)
    output = Activation('softmax')(x)

    model = Model(inputs=AE_output, outputs=output)
    return model


def combine(model_AE, model_CNN):
    RSS_train = Input(shape=(input_dim,), dtype='float32')
    AE_out = model_AE(RSS_train)
    AE_out = Reshape((size, size, 1))(AE_out)
    CNN_out = model_CNN(AE_out)
    model_AE_CNN = Model(outputs=CNN_out, inputs=RSS_train)
    model_AE_CNN.compile(loss=categorical_crossentropy, optimizer='adam', metrics=[metrics.categorical_accuracy])
    return model_AE_CNN


if __name__ == "__main__":
    backend.clear_session()
    args = param()
    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
    # building_weight = args.building_weight
    # floor_weight = args.floor_weight
    N = args.neighbours
    scaling = args.scaling
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    [RSS_train, RSS_val, y_train, y_val, dataset_name] = data_process()
    output_dim = y_train.shape[1]
    input_dim = RSS_train.shape[1]

    model_AE = build_model(sae_hidden_layers, input_dim, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS,
                           batch_size, epochs, VERBOSE, RSS_train)
    model_CNN = build_CNN(output_dim)
    model_AE_CNN = combine(model_AE, model_CNN)
    history = model_AE_CNN.fit(RSS_train, y_train, batch_size=batch_size, epochs=200, verbose=1,
                               validation_data=(RSS_val, y_val), shuffle=True)
    result = pd.DataFrame(history.history)
    result.to_csv(dataset_name + '_' + str(size) + '_acc_loss.csv')

    plt.figure()
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch = range(len(acc))

    plt.figure()
    plt.plot(epoch, acc, 'y', label='Training acc')
    plt.plot(epoch, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epoch, loss, 'y', label='Training loss')
    plt.plot(epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    del model_AE
    del model_CNN
    del model_AE_CNN
    backend.clear_session()
