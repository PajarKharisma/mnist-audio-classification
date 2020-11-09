import numpy as np
import matplotlib.pyplot as plt

def show_plot(**kwargs):
    data = kwargs
    plt.plot(data['train_data'])
    plt.plot(data['val_data'])
    plt.title(data['title'])
    plt.xlabel(data['xlabel'])
    plt.ylabel(data['ylabel'])
    plt.legend(['train', 'val'], loc='upper right')

    if data['should_save']:
        plt.savefig(data['path'])
    else:
        plt.show()
    plt.close()