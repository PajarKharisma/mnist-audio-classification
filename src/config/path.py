import os
import uuid

class Path():
    root_dir = os.getcwd()
    path = root_dir + '/dataset/'
    audio_file = path + 'audio/'
    csv_file = path + 'train.csv'