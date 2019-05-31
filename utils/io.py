import numpy as np
import os


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def load_sequence(sequence, root_folder='/home/datasets/typhoon/wnp/image/'):
    import h5py
    folder = os.path.join(root_folder, sequence+'/')
    files = sorted(os.listdir(folder))
    array = []
    for f in files:
        file = h5py.File(os.path.join(folder, f), 'r')
        img = file['infrared'][()]
        img -= img.min()
        eps = 1e-7
        array.append(2 * (img / (np.max(img) + eps)).astype(np.float32) - 1)
    return np.asarray(array)


def save_numpy(arr, path):
    create_folder(path)
    np.save(path, arr)


def get_typhoon_name(filename):
    return filename.split('-')[2].split('.h')[0]


