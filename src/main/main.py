import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from keras.utils import to_categorical

from src.config.path import *
from src.config.param import *
import src.utils.data_preparation as data_preparation
import src.utils.pre_process as pp
import src.utils.models as arch
import src.utils.visual as vis

def create_dataset():
    data_preparation.create_csv(Path.csv_file_train, Path.audio_file_train)
    data_preparation.create_csv(Path.csv_file_test, Path.audio_file_test)

def mfcc():
    df_train = pd.read_csv(Path.csv_file_train)
    x_train = []
    y_train = []
    for index, row in df_train.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_train + row['file'], res_type='kaiser_fast')
        emp_signal = pp.pre_emp_op(signal)
        feature = pp.extract_features(emp_signal, sample_rate)
        label = int(row['class'])

        x_train.append(feature)
        y_train.append(label)
    x_train = np.array(x_train)

    df_test = pd.read_csv(Path.csv_file_test)
    x_test = []
    y_test = []
    for index, row in df_test.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_test + row['file'], res_type='kaiser_fast')
        emp_signal = pp.pre_emp_op(signal)
        feature = pp.extract_features(emp_signal, sample_rate)
        label = int(row['class'])

        x_test.append(feature)
        y_test.append(label)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test

def audio_energy(norm=True):
    df_train = pd.read_csv(Path.csv_file_train)
    x_train = []
    y_train = []
    for index, row in df_train.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_train + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        energy = np.array([pp.audio_energy(frame) for frame in frames])
        if norm:
            energy = pp.norm_feature(energy)
        x_train.append(energy)

        label = int(row['class'])
        y_train.append(label)
    x_train = pp.padding(x_train, Param.features_dim)

    df_test = pd.read_csv(Path.csv_file_test)
    x_test = []
    y_test = []
    for index, row in df_test.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_test + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        energy = np.array([pp.audio_energy(frame) for frame in frames])
        if norm:
            energy = pp.norm_feature(energy)
        x_test.append(energy)

        label = int(row['class'])
        y_test.append(label)
    x_test = pp.padding(x_test, Param.features_dim)

    return x_train, y_train, x_test, y_test

def zero_crossing_rate(norm=True):
    df_train = pd.read_csv(Path.csv_file_train)
    x_train = []
    y_train = []
    for index, row in df_train.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_train + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        zcr_val = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            zcr_val = pp.norm_feature(zcr_val)
        x_train.append(zcr_val)

        label = int(row['class'])
        y_train.append(label)
    x_train = pp.padding(x_train, Param.features_dim)

    df_test = pd.read_csv(Path.csv_file_test)
    x_test = []
    y_test = []
    for index, row in df_test.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_test + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        zcr_val = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            zcr_val = pp.norm_feature(zcr_val)
        x_test.append(zcr_val)

        label = int(row['class'])
        y_test.append(label)
    x_test = pp.padding(x_test, Param.features_dim)

    return x_train, y_train, x_test, y_test

def entroy_of_energy(norm=True):
    df_train = pd.read_csv(Path.csv_file_train)
    x_train = []
    y_train = []
    for index, row in df_train.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_train + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        ee = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            ee = pp.norm_feature(ee)
        x_train.append(ee)

        label = int(row['class'])
        y_train.append(label)
    x_train = pp.padding(x_train, Param.features_dim)

    df_test = pd.read_csv(Path.csv_file_test)
    x_test = []
    y_test = []
    for index, row in df_test.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file_test + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        ee = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            ee = pp.norm_feature(ee)
        x_test.append(ee)

        label = int(row['class'])
        y_test.append(label)
    x_test = pp.padding(x_test, Param.features_dim)

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = mfcc()
    # x_train, y_train, x_test, y_test = audio_energy()
    # x_train, y_train, x_test, y_test = zero_crossing_rate()
    # x_train, y_train, x_test, y_test = entroy_of_energy()

    n_classes = len(set(y_train))
    y_train = to_categorical(y_train, dtype ="uint8")
    y_test = to_categorical(y_test, dtype ="uint8")

    model = arch.mlp(x_train.shape[1], n_classes)
    model.summary()
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=Param.epoch,
        batch_size=Param.batch_size,
        verbose=1
    )

    model.save_weights(Path.save_model)

    vis.show_plot(
        train_data=history.history['accuracy'],
        val_data=history.history['val_accuracy'],
        title='Model Accuracy',
        xlabel='epoch',
        ylabel='accuracy',
        should_save=True,
        path=Path.save_plot+'akurasi.png'
    )

    vis.show_plot(
        train_data=history.history['loss'],
        val_data=history.history['val_loss'],
        title='Model Loss',
        xlabel='epoch',
        ylabel='loss',
        should_save=True,
        path=Path.save_plot+'loss.png'
    )

if __name__ == "__main__":
    main()