from utils.tensors import *
import numpy as np
from scripts.latent_space import reverse_z


def forward(model, input_imgs, b=8, gpu=0):
    gen = batch_gen(input_imgs, batch_size=b)
    output = []
    model.eval()
    for arr in gen:
        print(arr.shape)
        tensor = convert_numpy_to_tensor(arr, gpu)
        print(tensor.size())
        reconstruct = model(tensor, only_decode=True)
        reconstruct_array = convert_tensor_to_numpy(reconstruct_array)
        output.append(reconstruct_array)
    return np.asarray(output)


def decoding(model, z, b=8, gpu=0):
    gen = batch_gen(z, batch_size=b, vector=True)
    output = []
    model.eval()
    for arr in gen:
        tens_out = model.decoder(convert_numpy_to_tensor(arr, gpu, vector=True))
        output.append(convert_tensor_to_numpy(tens_out))
    return np.asarray(output)


def encoding(model, arr, b=8, gpu=0, optimize_z=False, **kwargs):
    gen = batch_gen(arr, batch_size=b)
    z_size = model.z_size
    output = []
    model.eval()
    for arr in gen:
        goal_tens = convert_numpy_to_tensor(arr, gpu)
        tens_out = model.encoder(goal_tens)
        if optimize_z:
            tens_out = reverse_z(model.decoder, goal_tens, cuda=gpu,  z_size=z_size, **kwargs)
        output.append(convert_tensor_to_numpy(tens_out))
    return np.asarray(output)

