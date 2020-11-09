import os
import pandas as pd

def create_csv(dir_csv, dir_audio):
    files = os.listdir(dir_audio)
    dataset = []
    for file in files:
        data = {}
        data['file'] = file
        data['class'] = file.split('_')[0]
        dataset.append(data)

    df = pd.DataFrame(dataset)
    df.to_csv(dir_csv, index=False)
