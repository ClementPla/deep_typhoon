from utils.tensors import *
from scripts.latent_space import *
import torch


def deep_interpolation(arr1, arr2, nb_frames, model, optimize_z=True, spherical=True, **kwargs):
    model.eval()
    gpu = model.gpu
    z_size = model.z_size
    tens1 = convert_numpy_to_tensor(arr1, gpu)
    tens2 = convert_numpy_to_tensor(arr2, gpu)

    z1 = model.encoder(tens1)
    z2 = model.encoder(tens2)

    z1_norm = torch.norm(z1)
    z2_norm = torch.norm(z2)

    if optimize_z:
        z1 = reverse_z(model.decoder, tens1, z_size, gpu, initial_z=z1, **kwargs)
        z2 = reverse_z(model.decoder, tens2, z_size, gpu, initial_z=z2, **kwargs)

    interpolated_frames = []

    if spherical:
        theta = get_angle(z1, z2)
        z1 = z1/z1_norm
        z2 = z2 / z2_norm
        z_mean_norm = (z1_norm+z2_norm)/2

    time = np.linspace(0, 1, nb_frames+1)[1:]
    for t in time:
        if spherical:
            z = z1 * torch.sin((1 - t) * theta) / torch.sin(theta) + z2 * torch.sin(t * theta) / torch.sin(theta)
            z *= z_mean_norm
        else:
            z = z1*(1-t)+z2*t

        tens = model.decoder(z)

        interpolated_frames.append(convert_tensor_to_numpy(tens))

    return interpolated_frames


def linear_interpolation(arr1, arr2, nb_frames):
    interpolated_frames = []
    for f in range(nb_frames):
        f = f/nb_frames
        arr = arr1*(1-f)+arr2*f
        interpolated_frames.append(arr)
    return interpolated_frames



