from  os import path, makedirs


def create_folder(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)

