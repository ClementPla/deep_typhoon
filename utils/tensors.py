import numpy as np
import torch


def remove_nan(*tensor):
    for tens in tensor:
        tens[tens != tens] = 0


def check_nan(state_dict):
    for k in state_dict:
        if np.isnan(state_dict[k].numpy()).any():
            raise ValueError("Corrupted file")


def convert_numpy_to_tensor(arr, cuda=None):
    if arr.ndim == 2:
        arr = np.expand_dims(arr, 0)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)

    import torch
    if cuda is None:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(arr).cuda(cuda)


def convert_tensor_to_numpy(tensor):
    with torch.no_grad():
        return np.squeeze(tensor.cpu().numpy())

def apply_model(arr, model, cuda=None):
    if cuda is None:
        cuda = model.gpu

    return convert_tensor_to_numpy(model(convert_numpy_to_tensor(arr, cuda)))