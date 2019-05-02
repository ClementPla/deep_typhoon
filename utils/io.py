import numpy as np
import os


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files = [os.path.join(dirpath, f) for f in files]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    return os.path.join(dirpath, files[-1])


def save_numpy(arr, path):
    create_folder(path)
    np.save(path, arr)

