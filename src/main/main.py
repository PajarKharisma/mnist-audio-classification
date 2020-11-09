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

def main():
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

    n_classes = len(set(labels))
    input_dim = features[0].shape

    features = np.array(features)
    labels = to_categorical(labels, dtype ="uint8")

    model = arch.mlp(features.shape[1], n_classes)
    model.summary()
    history = model.fit(
        features,
        labels,
        validation_split=Param.val_split,
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

    # for label, feature in zip(labels, features):
    #     print('{} --> {}'.format(feature, label))

if __name__ == "__main__":
    main()