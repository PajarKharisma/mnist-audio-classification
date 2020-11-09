import sys
import os
import time
from datetime import datetime
root_dir = os.getcwd()
sys.path.append(root_dir)

from src.config.path import *

import src.utils.data_preparation as data_preparation

def create_dataset():
    data_preparation.create_csv(Path.csv_file, Path.audio_file)

create_dataset()