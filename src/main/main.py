import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

import numpy as np
import pandas as pd
import librosa.display
import librosa

from keras.utils import to_categorical

from src.config.path import *
from src.config.param import *
import src.utils.data_preparation as data_preparation

def create_dataset():
    data_preparation.create_csv(Path.csv_file, Path.audio_file)

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    duration = len(audio)/sample_rate
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20,
        hop_length=int(Param.window_stride * sample_rate),
        n_fft=int(Param.window_size * sample_rate)
    )
    delta = librosa.feature.delta(mfccs)
    deltad = librosa.feature.delta(mfccs, order=2)

    mfccs = np.mean(mfccs.T, axis=0)
    delta = np.mean(delta.T, axis=0)
    deltad = np.mean(deltad.T, axis=0)

    return np.concatenate((mfccs, delta, deltad), axis=0) 

def main():
    df = pd.read_csv(Path.csv_file)
    features = []
    labels = []
    for index, row in df.iterrows():
        feature = extract_features(Path.audio_file + row['file'])
        label = int(row['class'])

        features.append(feature)
        labels.append(label)
    
    labels = to_categorical(labels, dtype ="uint8") 
    for label, feature in zip(labels, features):
        print('{} --> {}'.format(feature, label))

if __name__ == "__main__":
    main()