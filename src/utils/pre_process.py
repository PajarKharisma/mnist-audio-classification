from src.config.param import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

import librosa
import numpy as np

def windowing(signal, sample_rate, frame_size, frame_stride):
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    frames *= np.hamming(frame_length)
    return frames

def audio_energy(frame):
    return np.sum(np.power(frame,2)) / len(frame)

def zero_crossing_rate(frame):
    zcr = 0
    for i in range(1, len(frame)):
        zcr += abs(np.sign(frame[i]) - np.sign(frame[i-1]))
    return zcr / (2*len(frame))

def entropy_of_energy(frame, split=10):
    sub_frame = np.array_split(frame, split)
    e_sub_frame = np.array([audio_energy(sf) for sf in sub_frame])
    e_short_frame = np.sum(e_sub_frame)
    ej = e_sub_frame / e_short_frame
    return (-1) * np.sum(ej * np.nan_to_num(np.log2(ej)))

def norm_feature(data):
    variance = np.var(data)
    mean = np.mean(data)
    return (data - variance) / mean

def pre_emp_op(signal):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def extract_features(signal, sample_rate):
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40,
        hop_length=int(Param.window_stride * sample_rate),
        n_fft=int(Param.window_size * sample_rate)
    )
    delta = librosa.feature.delta(mfccs)
    deltad = librosa.feature.delta(mfccs, order=2)

    mfccs = np.mean(mfccs.T, axis=0)
    delta = np.mean(delta.T, axis=0)
    deltad = np.mean(deltad.T, axis=0)

    return np.concatenate((mfccs, delta, deltad), axis=0) 
    # return mfccs.reshape(-1)

def padding(features, dim):
    return pad_sequences(
        features,
        maxlen=dim,
        padding='post',
        truncating='post',
        dtype='float32'
    )