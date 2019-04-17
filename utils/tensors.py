import numpy as np
import torch
import math

def remove_nan(*tensor):
    for tens in tensor:
        tens[tens != tens] = 0


def check_nan(state_dict):
    for k in state_dict:
        if np.isnan(state_dict[k].numpy()).any():
            raise ValueError("Corrupted file")


def convert_numpy_to_tensor(arr, cuda=None, vector=False, expect_dims=3):
    if not vector:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        if arr.ndim == 3 and expect_dims != 3:
            arr = np.expand_dims(arr, 0)
    if vector:
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)

    print(arr.shape)
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


def batch_gen(arr, vector=False, batch_size=8):

    if not vector:
        if arr.ndim in [2, 3]:
            yield arr
        if arr.shape[0] < 8:
            yield arr
    dims = arr.shape[0]
    for i in range(math.ceil(dims/batch_size)):
        yield arr[i*batch_size:(i+1)*batch_size]

