import argparse
import os

import numpy as np
import pandas as pd
import tensorflow.keras.backend as backend
from tensorflow.keras import regularizers, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Reshape, BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, \
    Activation, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Adam
from process import data_process

# import matplotlib.pyplot as plt

#tensorboard = TensorBoard(log_dir='./newlogs', histogram_freq=1, write_graph=True, write_images=True)
current_dir = os.path.dirname(os.path.abspath(__file__))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
EARLYSTOP = 0
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
SAE_EPOCH = 100
SAE_BATCHSIZE = 1024
# ------------------------------------------------------------------------
# CNN
# ------------------------------------------------------------------------
CNN_ACTIVATION = 'relu'
CNN_BIAS = False
LOSS = categorical_crossentropy
EPOCH = 100
OPTIMIZER = 'adam'
CNN_BATCHSIZE = 1024
filter1 = 34
filter2 = 64
dropout = 0.1
learning_rate = 0.0005

def param():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--log_dir',
    #     type=str,
    #     default='./logs',
    #     help='Summaries log directory')
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
    # parser.add_argument(
    #     "-E",
    #     "--epochs",
    #     help="number of epochs; default is 50",
    #     default=50,
    #     type=int)
    # parser.add_argument(
    #     "-B",
    #     "--batch_size",
    #     help="batch size; default is 10",
    #     default=256,
    #     type=int)
    # parser.add_argument(
    #     "-T",
    #     "--training_ratio",
    #     help="ratio of training data to overall data: default is 0.90",
    #     default=0.9,
    #     type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        # default='256, 128,' + str(hidden_nodes) + ',128, 256',
        default='510, 0, 510',
        type=str)
    # parser.add_argument(
    #     "-C",
    #     "--classifier_hidden_layers",
    #     help=
    #     "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
    #     default='128,128',
    #     type=str)
    # parser.add_argument(
    #     "-D",
    #     "--dropout",
    #     help=
    #     "dropout rate before and after classifier hidden layers; default 0.0",
    #     default=0.1,
    #     type=float)
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
    # parser.add_argument(
    #     "-N",
    #     "--neighbours",
    #     help="number of (nearest) neighbour locations to consider in positioning; default is 1",
    #     default=1,
    #     type=int)
    # parser.add_argument(
    #     "--scaling",
    #     help=
    #     "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
    #     default=0.0,
    #     type=float)
    args = parser.parse_args()
    return args


# create the autoencoder
def build_AE(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS, SAE_BATCHSIZE):
    sae_reduce_lr = ReduceLROnPlateau(optimizer=SAE_OPTIMIZER, monitor='val_loss', patience=10, factor=0.5, mode='min',
                                      min_lr=1e-5)
    model = Sequential()
    # model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activity_regularizer=regilization))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    for units in sae_hidden_layers[1:]:
        model.add(Dense(units, activation=SAE_ACTIVATION, activity_regularizer=regilization, use_bias=SAE_BIAS, ))
        # model.add(Dense(units,activity_regularizer=regilization))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
    # model.add(Dense(INPUT_DIM, activity_regularizer=regilization))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
    model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS, metrics=['acc'])
    # train the model
    if (EARLYSTOP is 1):
        model.fit(x_train, x_train, batch_size=SAE_BATCHSIZE, epochs=SAE_EPOCH, verbose=VERBOSE, shuffle=True,
                  validation_data=(x_val, x_val), callbacks=[early_stopping])
    else:
        model.fit(x_train, x_train, batch_size=SAE_BATCHSIZE, epochs=SAE_EPOCH, verbose=VERBOSE, shuffle=True,
                  validation_data=(x_val, x_val), callbacks=[sae_reduce_lr])
    # remove the decoder part
    num_to_remove = (len(sae_hidden_layers) + 1) // 2
    # num_to_remove = 6
    for i in range(num_to_remove):
        model.pop()
    model.add(Dropout(rate=dropout))
    del sae_reduce_lr
    return model


def build_CNN(OUTPUT_DIM):
    AE_output = Input(shape=(SIZE, SIZE, 1))
    x = AE_output
    x = Conv2D(filter1, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout)(x)
    x = Flatten()(x)
    x = Dense(400)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
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
    model_AE_CNN.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=[metrics.categorical_accuracy])
    return model_AE_CNN


def train_AE_CNN(model_AE_CNN, model_AE):
    #optimizer = optimizers.Adam(lr=0.0001)
    # reduce_lr = ReduceLROnPlateau(optimizer=optimizer, monitor='val_loss', patience=10, factor=LR_REDUCE_FACTOR, mode='min', min_lr=1e-9)
    AE_output = model_AE.predict(x_test)
    output_path = root_folder + "//AE_output//pre//test//" + str(SIZE) + ".csv"

    pd.DataFrame(AE_output).to_csv(output_path)

    if EARLYSTOP is 1:
        history = model_AE_CNN.fit(x_train, y_train, batch_size=CNN_BATCHSIZE, epochs=EPOCH, verbose=VERBOSE,
                                   validation_data=(x_val, y_val), shuffle=True, callbacks=[early_stopping])
    else:
        history = model_AE_CNN.fit(x_train, y_train, batch_size=CNN_BATCHSIZE, epochs=EPOCH, verbose=VERBOSE,
                                   validation_data=(x_val, y_val), shuffle=True)
    AE_output = model_AE.predict(x_test)
    output_path = root_folder + "//AE_output//after//test//" + str(SIZE) + ".csv"
    pd.DataFrame(AE_output).to_csv(output_path)
    AE_CNN_output = model_AE_CNN.predict(x_test)
    hit_num = 0
    for i in range(0, AE_CNN_output.shape[0]):
        if np.argmax(AE_CNN_output, axis=1)[i] == np.argmax(y_test, axis=1)[i]:
            hit_num += 1
            # true_index.append(i)
        # else:
        # false_index.append(i)
    hr = hit_num / AE_CNN_output.shape[0]
    hr_ls.append(hr)
    #del optimizer
    #del reduce_lr
    return history


if __name__ == "__main__":
    args = param()
    # set variables using command-line arguments
    gpu_id = args.gpu_id
    # random_seed = args.random_seed
    # epochs = args.epochs
    # batch_size = args.batch_size
    # training_ratio = args.training_ratio
    # if args.classifier_hidden_layers == '':
    #     classifier_hidden_layers = ''
    # else:
    #     classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    # dropout = args.dropout
    # building_weight = args.building_weight
    # floor_weight = args.floor_weight
    # N = args.neighbours
    # scaling = args.scaling
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    [x_trainval, y_trainval, x_train, y_train, x_val, y_val, x_test, y_test, dataset_name] = data_process()
    OUTPUT_DIM = y_train.shape[1]
    INPUT_DIM = x_train.shape[1]
    # pd.DataFrame(x_train).to_csv('preprocessed_data//' + 'train' + '_' + '.csv')
    # pd.DataFrame(x_val).to_csv('preprocessed_data//' + 'val' + '_' + '.csv')
    # pd.DataFrame(x_test).to_csv('preprocessed_data//' + 'test' + '_' + '.csv')

    for dropout in [0.1]:
        #for LR_REDUCE_FACTOR in [0.05, 0.1, 0.5, 0.9]:
        for LR_REDUCE_FACTOR in [1]:
            hr_ls = []
            for SIZE in range(6, 23, 2):
                root_folder = 'd' + str(dropout) + 'f' + str(LR_REDUCE_FACTOR) + '//' + str(
                    SIZE) + 'x' + str(SIZE)
                # true_index = []
                # false_index = []
                sae_hidden_layers = [int(i) for i in args.sae_hidden_layers.split(',')]
                sae_hidden_layers[1] = SIZE * SIZE
                model_AE = build_AE(sae_hidden_layers, INPUT_DIM, SAE_ACTIVATION, SAE_BIAS, SAE_OPTIMIZER, SAE_LOSS,
                                    SAE_BATCHSIZE)
                model_CNN = build_CNN(OUTPUT_DIM)
                model_AE_CNN = combine_models(model_AE, model_CNN)
                history = train_AE_CNN(model_AE_CNN, model_AE)
                result = pd.DataFrame(history.history)
                result.to_csv(root_folder + '//acc_loss_result//' + dataset_name + '_' + str(SIZE) + '_acc_loss.csv')
                # pd.DataFrame(true_index).to_csv(
                #     root_folder + '//acc_loss_result//' + dataset_name + '_' + str(SIZE) + '_test_true_index.csv')
                # pd.DataFrame(false_index).to_csv(
                #     root_folder + '//acc_loss_result//' + dataset_name + '_' + str(SIZE) + '_test_false_index.csv')
                del model_AE
                del model_CNN
                del model_AE_CNN
                backend.clear_session()
                ops.reset_default_graph()
            name = ['hitting_rate']
            hr_df = pd.DataFrame(columns=name, data=hr_ls)
            dir = os.path.join(current_dir,
                               "d" + str(dropout) + '_' + "f" + str(LR_REDUCE_FACTOR) + '_' + 'test_acc' + '.csv')
            hr_df.to_csv(dir)

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
