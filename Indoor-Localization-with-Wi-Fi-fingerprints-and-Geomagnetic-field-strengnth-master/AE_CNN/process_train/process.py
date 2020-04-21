import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# file_list = ['XJTLU/Dataset_4thFloor_MIX2.csv', 'XJTLU/Dataset_4thFloor_HuaWei.csv']
file_list = ['UJIIndoorLoc/trainingData2.csv']


def data_process():
    if len(file_list) == 1:
        select_AP, select_label, dataset_name = UJI()
    else:
        select_AP, select_label, dataset_name = XJTLU()
    x_trainval, x_test, y_trainval, y_test = train_test_split(select_AP, select_label, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.11, random_state=42)
    return [x_trainval, y_trainval, x_train, y_train, x_val, y_val, x_test, y_test, dataset_name]


def XJTLU(data_size=516):
    data = []
    for file in file_list:
        df = pd.read_csv(file, index_col=0)
        df = df.drop(
            columns=['Floor', 'Building', 'Model', 'GeoX', 'GeoY', 'GeoZ', 'AccX', 'AccY', 'AccZ', 'OriX', 'GriY',
                     'GriZ', 'rGeoX', 'rGeoY', 'rGeoZ', 'rAccX', 'rAccY', 'rAccZ'])
        data.append(df)
    df = pd.concat([data[0], data[1]], ignore_index=True)
    arr = np.asarray(df)
    AP = arr[:, 0:data_size]
    x_all = arr[:, data_size:data_size + 1]
    y_all = arr[:, data_size + 1:]
    labels = np.zeros(x_all.shape[0])
    for i in range(0, x_all.shape[0]):
        labels[i] = (y_all[i] * 51 + x_all[i] + 1)
    onehot_encoder = OneHotEncoder()
    train_labels = onehot_encoder.fit_transform(labels.reshape(x_all.shape[0], 1)).toarray()
    scaler = StandardScaler()
    AP = AP / -110.0
    AP = scaler.fit_transform(AP.astype(float))
    all_data = np.concatenate([AP, train_labels], axis=1)
    np.random.shuffle(all_data)
    select_AP = all_data[:, 0:data_size]
    select_label = all_data[:, data_size:]
    dataset_name = 'XJTLU'
    return select_AP, select_label, dataset_name


def UJI(data_size=520):
    for file in file_list:
        df = pd.read_csv(file)
    arr = np.asarray(df)
    AP = arr[:, 0:data_size]
    y_all = arr[:, data_size:]
    onehot_encoder = OneHotEncoder()
    train_labels = onehot_encoder.fit_transform(y_all.reshape(y_all.shape[0], 1)).toarray()
    scaler = StandardScaler()
    AP = AP / -110.0
    AP = scaler.fit_transform(AP.astype(float))
    all_data = np.concatenate([AP, train_labels], axis=1)
    np.random.shuffle(all_data)
    # select_data = np.asarray(random.sample(list(all_data), 2000))
    select_AP = all_data[:, 0:data_size]
    select_label = all_data[:, data_size:]
    dataset_name = 'UJI'
    return select_AP, select_label, dataset_name
