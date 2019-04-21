from os import path, makedirs
import numpy as np

def create_folder(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)


def save_numpy(arr, path):
    create_folder(path)
    np.save(path, arr)

