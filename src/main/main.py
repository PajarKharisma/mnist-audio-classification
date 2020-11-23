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
    data_preparation.create_csv(Path.csv_file, Path.audio_file)

def mfcc():
    df = pd.read_csv(Path.csv_file)
    features = []
    labels = []
    for index, row in df.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file + row['file'], res_type='kaiser_fast')
        emp_signal = pp.pre_emp_op(signal)
        feature = pp.extract_features(emp_signal, sample_rate)
        label = int(row['class'])

        features.append(feature)
        labels.append(label)

    features = np.array(features)

    return feature, labels

def audio_energy(norm=True):
    df = pd.read_csv(Path.csv_file)
    features = []
    labels = []
    
    for index, row in df.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        energy = np.array([pp.audio_energy(frame) for frame in frames])
        if norm:
            energy = pp.norm_feature(energy)
        features.append(energy)

        label = int(row['class'])
        labels.append(label)

    features = pp.padding(features, Param.features_dim)

    return features, labels

def zero_crossing_rate(norm=True):
    df = pd.read_csv(Path.csv_file)
    features = []
    labels = []
    for index, row in df.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        zcr_val = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            zcr_val = pp.norm_feature(zcr_val)
        features.append(zcr_val)

        label = int(row['class'])
        labels.append(label)

    features = pp.padding(features, Param.features_dim)

    return features, labels

def entroy_of_energy(norm=True):
    df = pd.read_csv(Path.csv_file)
    features = []
    labels = []
    for index, row in df.iterrows():
        signal, sample_rate = librosa.load(Path.audio_file + row['file'], res_type='kaiser_fast')
        frames = pp.windowing(signal, sample_rate, Param.window_size, Param.window_stride)
        ee = np.array([pp.zero_crossing_rate(frame) for frame in frames])
        if norm:
            ee = pp.norm_feature(ee)
        features.append(ee)

        label = int(row['class'])
        labels.append(label)

    features = pp.padding(features, Param.features_dim)

    return features, labels

def main():
    # features, labels = mfcc()
    # features, labels = audio_energy()
    # features, labels = zero_crossing_rate()
    features, labels = entroy_of_energy()

    n_classes = len(set(labels))
    labels = to_categorical(labels, dtype ="uint8")

    model = arch.mlp(features.shape[1], n_classes)
    model.summary()
    history = model.fit(
        x=features,
        y=labels,
        validation_data=(features, labels),
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