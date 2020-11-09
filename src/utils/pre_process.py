from src.config.param import *

import librosa
import numpy as np

def pre_emp_op(signal):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def extract_features(signal, sample_rate):
    mfccs = librosa.feature.mfcc(
        y=signal,
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