import os
import uuid

class Path():
    root_dir = os.getcwd()
    path = root_dir + '/dataset/'
    audio_file_train = path + 'train/'
    csv_file_train = path + 'train.csv'

    audio_file_test = path + 'test/'
    csv_file_test = path + 'test.csv'

    save_plot = root_dir + '/log/plot/'
    save_model = '{}/models/{}.h5'.format(root_dir, str(uuid.uuid4().hex))