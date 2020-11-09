import os
import uuid

class Path():
    root_dir = os.getcwd()
    path = root_dir + '/dataset/'
    audio_file = path + 'audio/'
    csv_file = path + 'train.csv'

    save_plot = root_dir + '/log/plot/'
    save_model = '{}/models/{}.h5'.format(root_dir, str(uuid.uuid4().hex))