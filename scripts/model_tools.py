from utils.tensors import *
import numpy as np
from scripts.latent_space import reverse_z
from threading import Thread
from os import makedirs, path

def forward(model, input_imgs, b=8, gpu=0, optimize_z=False, **kwargs):
    gen = batch_gen(input_imgs, batch_size=b)
    output = []
    model.eval()
    for arr in gen:
        tensor = convert_numpy_to_tensor(arr, gpu)
        # reconstruct = model(tensor, only_decode=True)
        z = model.encoder(tensor)
        if optimize_z:
            z = reverse_z(model.decoder, tensor, cuda=gpu, z_size=model.z_size, initial_z=z, **kwargs)
        reconstruct = model.decoder(z)
        reconstruct_array = convert_tensor_to_numpy(reconstruct, squeeze=False)
        output.append(reconstruct_array)
    return np.concatenate(output, axis=0)


def decoding(model, z, b=8, gpu=0):
    gen = batch_gen(z, batch_size=b, vector=True)
    output = []
    model.eval()
    for arr in gen:
        tens_out = model.decoder(convert_numpy_to_tensor(arr, gpu, vector=True))
        output.append(convert_tensor_to_numpy(tens_out))
    return np.concatenate(output, 0)


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
    return np.concatenate(output, 0)


class MultiThreadingEncoder(Thread):
    def __init__(self, model, gpu, dataset, folder, **options):

        super(MultiThreadingEncoder, self).__init__()
        self.model = model
        self.gpu = gpu
        self.dataset = dataset
        self.options = options
        self.folder = folder
        if not path.exists(folder):
            makedirs(folder)

    def run(self):
        for sequence in self.dataset:
            name = sequence['name']
            arr = sequence['data']
            z = encoding(self.model, arr, self.gpu, **self.options)
            np.save(path.join(self.folder, name+'.npy'), z)




