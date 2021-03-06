from utils.tensors import *
from scripts.latent_space import *
from networks.VAE import *
import torch
import cv2


def deep_interpolation(arr1, arr2, nb_frames, model, optimize_z=True, spherical=True, **kwargs):
    model.eval()
    gpu = model.gpu
    z_size = model.z_size
    tens1 = convert_numpy_to_tensor(arr1, gpu)
    tens2 = convert_numpy_to_tensor(arr2, gpu)

    if isinstance(model, VaeGan):
        z1 = model.encode(tens1)
        z2 = model.encode(tens2)
    else:
        z1 = model.encoder(tens1)
        z2 = model.encoder(tens2)

    if optimize_z:
        z1 = reverse_z(model.decoder, tens1, z_size, gpu, initial_z=z1, **kwargs)
        z2 = reverse_z(model.decoder, tens2, z_size, gpu, initial_z=z2, **kwargs)

    return latent_space_interpolation(z1, z2, decoder=model.decoder, nb_frames=nb_frames, spherical=spherical)


def latent_space_interpolation(z1, z2, decoder, nb_frames, spherical=True):
    z1_norm = torch.norm(z1)
    z2_norm = torch.norm(z2)
    interpolated_frames = []

    if spherical:
        theta = get_angle(z1, z2)
        z1 = z1 / z1_norm
        z2 = z2 / z2_norm
        z_mean_norm = (z1_norm + z2_norm) / 2

    time = np.linspace(0, 1, nb_frames + 1, endpoint=False)[1:]
    for t in time:
        if spherical:
            z = z1 * torch.sin((1 - t) * theta) / torch.sin(theta) + z2 * torch.sin(t * theta) / torch.sin(theta)
            z *= z_mean_norm
        else:
            z = z1 * (1 - t) + z2 * t

        tens = decoder(z, only_last=True)

        interpolated_frames.append(convert_tensor_to_numpy(tens))
    return interpolated_frames


def polar_interpolation(arr1, arr2, nb_frames, return_polar=False):
    value = np.sqrt(((arr1.shape[0] / 2.0) ** 2.0) + ((arr1.shape[1] / 2.0) ** 2.0))
    polar_image1 = cv2.linearPolar(arr1, (arr1.shape[0] / 2, arr1.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
    polar_image2 = cv2.linearPolar(arr2, (arr1.shape[0] / 2, arr1.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
    time = np.linspace(0, 1, nb_frames + 1, endpoint=False)[1:]
    interpolated_frames = []
    interpolated_polar = []
    for t in time:
        polar_img = polar_image1 * (1 - t) + polar_image2 * t
        cartesian_image = cv2.linearPolar(polar_img, (arr1.shape[0] / 2, arr1.shape[1] / 2), value,
                                          cv2.WARP_INVERSE_MAP)

        cartesian_image = np.clip(cartesian_image, arr1.min(), arr2.max())

        interpolated_frames.append(cartesian_image)
        interpolated_polar.append(polar_img)
    if return_polar:
        return interpolated_frames, interpolated_polar
    else:
        return interpolated_frames


def linear_interpolation(arr1, arr2, nb_frames):
    interpolated_frames = []
    time = np.linspace(0, 1, nb_frames + 1, endpoint=False)[1:]
    for t in time:
        arr = arr1 * (1 - t) + arr2 * t
        interpolated_frames.append(arr)
    return interpolated_frames
