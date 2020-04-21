import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_image_size():
    x_axis = 'size of image'
    dict = {}
    for i, e in enumerate(size_ls):
        dict[e] = metrics_to_size[i]
    frame = [(k, dict[k]) for k in sorted(dict.keys())]
    plt.plot(*zip(*frame))
    plt.title(y_axis + '-' + x_axis)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    for a, b in zip(*zip(*frame)):
        plt.text(a, b, '%0.3f' % (b), ha='center', va='bottom', fontsize=10)
    plt.show()


def plot_epoch():
    x_axis = 'epoch'
    frame = metrics_to_epoch[0]
    for i, e in enumerate(metrics_to_epoch):
        if i != len(metrics_to_epoch) - 1:
            frame = pd.concat([frame, metrics_to_epoch[i + 1]], axis=1)
    frame.plot(kind='line')
    plt.title(y_axis + '-' + x_axis)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


def import_data(y_axis):
    metrics_to_epoch = []
    metrics_to_size = []
    size_ls = []
    for root, dirs, files in os.walk('acc_loss_result//', topdown=True):
        for name in files:
            filename, type = os.path.splitext(name)
            size = int(filename.split('_')[1])
            if type == '.csv':
                size_ls.append(size)
                path = os.path.join(root, name)
                df = pd.read_csv(path, index_col=0)

                if ('loss' in y_axis):
                    metrics_to_size.append(float(np.asarray(df[y_axis].min())))
                    if ('val_loss' is y_axis):
                        metrics_to_epoch.append(
                            df['val_loss'].to_frame().set_axis([str(size)], axis='columns', inplace=False))
                    else:
                        metrics_to_epoch.append(
                            df['loss'].to_frame().set_axis([str(size)], axis='columns', inplace=False))

                else:
                    metrics_to_size.append(float(np.asarray(df[y_axis].max())))
                    if ('val_categorical_accuracy' is y_axis):
                        metrics_to_epoch.append(
                            df['val_categorical_accuracy'].to_frame().set_axis([str(size)], axis='columns',
                                                                               inplace=False))
                    else:
                        metrics_to_epoch.append(
                            df['categorical_accuracy'].to_frame().set_axis([str(size)], axis='columns',
                                                                           inplace=False))

    return metrics_to_epoch, metrics_to_size, size_ls


if __name__ == '__main__':
    for y_axis in ['val_loss', 'val_categorical_accuracy', 'loss', 'categorical_accuracy']:
        metrics_to_epoch, metrics_to_size, size_ls = import_data(y_axis)
        plot_epoch()
        plot_image_size()
