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


def init_cuda_sequences_batch(input_tensor, max_batch_length, gpu):
    out_tens = []
    for tens in input_tensor:
        out_tens.append(tens.cuda(gpu)[:,:max_batch_length])
    return tuple(out_tens)


def convert_numpy_to_tensor(arr, cuda=None, vector=False, expect_dims=None):
    if not vector:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        if arr.ndim == 3 and expect_dims != 3:
            arr = np.expand_dims(arr, 0)
    if vector:
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)

    import torch
    if cuda is None:
        return torch.from_numpy(arr)
    else:
        return torch.from_numpy(arr).cuda(cuda)


def convert_tensor_to_numpy(tensor, squeeze=True):
    with torch.no_grad():
        if squeeze:
            return np.squeeze(tensor.cpu().numpy())
        else:
            return tensor.cpu().numpy()

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
        else:
            dims = arr.shape[0]
            for i in range(math.ceil(dims/batch_size)):
                yield arr[i*batch_size:(i+1)*batch_size]
    else:
        dims = arr.shape[0]
        for i in range(math.ceil(dims / batch_size)):
            yield arr[i * batch_size:(i + 1) * batch_size]


