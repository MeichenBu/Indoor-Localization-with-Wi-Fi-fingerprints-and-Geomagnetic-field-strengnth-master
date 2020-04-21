import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from keras import regularizers, optimizers, metrics
import keras.backend as backend
from keras.models import Sequential, Model
from keras.layers import  Input, Reshape, BatchNormalization,Conv2D, MaxPooling2D,Dense, Dropout, Activation, Flatten
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn import preprocessing
from tensorflow.python.framework import ops
from process import data_process
early_stopping = EarlyStopping(monitor='val_loss', patience= 10, verbose=2)

EARLYSTOP = 0
l1 = regularizers.l1(0.0)
l2 = regularizers.l2(0.01)
regilization = l1
VERBOSE = 1  # 0 for turning off logging
BATCHSIZE = 256
# ------------------------------------------------------------------------
# stacked auto encoder (sae)
# ------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
SAE_EPOCH = 50
SAE_BATCHSIZE = 256
# ------------------------------------------------------------------------
# CNN
# ------------------------------------------------------------------------
CNN_ACTIVATION = 'relu'
CNN_BIAS = False
LOSS = categorical_crossentropy
EPOCH = 100
OPTIMIZER = 'adam'
CNN_BATCHSIZE = 256

def param():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    # parser.add_argument(
    #     "-R",
    #     "--random_seed",
    #     help="random seed",
    #     default=0,
    #     type=int)
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
        default=256,
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
        default='500,' + str(hidden_nodes) + ',500',
        type=str)
    # parser.add_argument(
    #     "-C",
    #     "--classifier_hidden_layers",
    #     help=
    #     "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
    #     default='128,128',
    #     type=str)
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

# create the autoencoder
def build_AE(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS, SAE_BATCHSIZE):
    model = Sequential()
    model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    for units in sae_hidden_layers[1:]:
        model.add(Dense(units, activation=SAE_ACTIVATION, activity_regularizer=regilization, use_bias=SAE_BIAS,))
    model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS, metrics=['acc', 'loss'])
    # train the model
    if(EARLYSTOP is 1):
        model.fit(x_train, x_train, batch_size=SAE_BATCHSIZE, epochs=SAE_EPOCH, verbose=VERBOSE, shuffle=True,
                   validation_data=(x_val, x_val), callbacks=[early_stopping])
    else:
        model.fit(x_train, x_train, batch_size=SAE_BATCHSIZE, epochs=SAE_EPOCH, verbose=VERBOSE, shuffle=True,
              validation_data=(x_val, x_val))
    # remove the decoder part
    num_to_remove = (len(sae_hidden_layers) + 1) // 2
    for i in range(num_to_remove):
        model.pop()
    return model


def build_CNN(OUTPUT_DIM):
    AE_output = Input(batch_shape=(CNN_BATCHSIZE, SIZE, SIZE, 1))
    x = AE_output
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.4)(x)
    x = Flatten()(x)
    x = Dense(400)(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    x = Dense(OUTPUT_DIM)(x)
    x = BatchNormalization()(x)
    output = Activation('softmax')(x)
    model = Model(inputs=AE_output, outputs=output)
    return model


def combine_models(model_AE, model_CNN):
    x_train = Input(shape=(INPUT_DIM,), dtype='float32')
    AE_out = model_AE(x_train)
    CNN_input = Reshape((SIZE, SIZE, 1))(AE_out)
    CNN_output = model_CNN(CNN_input)
    model_AE_CNN = Model(outputs=CNN_output, inputs=x_train)
    model_AE_CNN.compile(loss=LOSS, optimizer= OPTIMIZER, metrics=[metrics.categorical_accuracy])
    return model_AE_CNN


def train_AE_CNN(model_AE_CNN, model_AE):
    optimizer = optimizers.Adam(lr=0.001, beta_1=0.99999, beta_2=0.999, epsilon=1e-09)
    reduce_lr = ReduceLROnPlateau(optimizer=optimizer, monitor='val_loss', patience=4, factor=0.9, mode='min')
    if (EARLYSTOP is 1):
        history = model_AE_CNN.fit(x_train, y_train, batch_size= BATCHSIZE, epochs= EPOCH, verbose=VERBOSE,
                               validation_data=(x_val, y_val), shuffle=True,  callbacks=[early_stopping, reduce_lr])
    else:
        history = model_AE_CNN.fit(x_train, y_train, batch_size= BATCHSIZE, epochs= EPOCH, verbose=VERBOSE,
                               validation_data=(x_val, y_val), shuffle=True, callbacks=[reduce_lr])
    AE_output = model_AE.predict(x_train)
    output_path = "AE_output//" + str(SIZE) + ".csv"
    pd.DataFrame(AE_output).to_csv(output_path)
    del optimizer
    del reduce_lr
    return history


if __name__ == "__main__":
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

    for SIZE in range(6, 23, 1):
        hidden_nodes = SIZE * SIZE
        [x_trainval, y_trainval, x_train, y_train, x_val, y_val, x_test, y_test, dataset_name] = data_process()
        OUTPUT_DIM = y_train.shape[1]
        INPUT_DIM = x_train.shape[1]

        model_AE = build_AE(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS, SAE_BATCHSIZE)
        model_CNN = build_CNN(OUTPUT_DIM)
        model_AE_CNN= combine_models(model_AE, model_CNN)

        history = train_AE_CNN(model_AE_CNN, model_AE)
        result = pd.DataFrame(history.history)
        result.to_csv('acc_loss_result//' + dataset_name + '_' + str(SIZE) + '_acc_loss.csv')

        del model_AE
        del model_CNN
        del model_AE_CNN

        backend.clear_session()
        ops.reset_default_graph()

        # ------------------------------------------------------------------------
        # plot acc and loss curves
        # ------------------------------------------------------------------------
        # acc = history.history['categorical_accuracy']
        # val_acc = history.history['val_categorical_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epoch = range(len(acc))

        # plt.figure()
        # plt.plot(epoch, acc, 'y', label='Training Acc')
        # plt.plot(epoch, val_acc, 'b', label='Validation Acc')
        # plt.title('Training and Validation Accuracy'+ ' Size = ' + str(size) + 'x' + str(size))
        # plt.legend()
        # plt.savefig('acc_loss_result//'+ dataset_name + '_' + str(size) + '_acc.png')

        # plt.figure()
        # plt.plot(epoch, loss, 'y', label='Training Loss')
        # plt.plot(epoch, val_loss, 'b', label='Validation Loss')
        # plt.title('Training and Validation Loss' + ' Size = ' + str(size) + 'x' + str(size))
        # plt.legend()
        # plt.savefig('acc_loss_result//''+ dataset_name + '_' + str(size) + '_loss.png')
        # plt.show()




