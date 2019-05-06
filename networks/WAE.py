import torch
import torch.nn as nn
import numpy
from .abstract_network import AbstractNet
from .basis_block import *
from utils.tensors import *


class Encoder(nn.Module):
    def __init__(self, channel_in=1, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        layers_list += [nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, padding=2, stride=1,
                                  bias=False), nn.BatchNorm2d(num_features=32, momentum=0.9)]
        for i in range(4):
            layers_list.append(EncoderBlock(channel_in=32 * (2 ** i), channel_out=32 * 2 ** (i + 1)))

        self.size = 32 * 2 ** (i + 1)

        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size))

        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=z_size),
                                nn.Sigmoid())

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        return ten

class Discriminator(nn.Module):
    def __init__(self, n_channel, dim_h, dim_z):
        super(Discriminator, self).__init__()

        self.n_channel = n_channel
        self.dim_h = dim_h
        self.n_z = dim_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x


class WAE(AbstractNet):
    def __init__(self, channel_in=1, z_size=128, dim_h=128, gpu=1, checkpoint='', upsampling='nearest'):
        super(WAE, self).__init__(gpu=gpu, checkpoint=checkpoint, upsampling=upsampling)
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size, channel_in=channel_in)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size, upsampling=upsampling)
        self.discriminator = Discriminator(channel_in, dim_h=dim_h, dim_z=z_size)
        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, ten, only_decode=False):
        if self.training:
            # encode
            ten_encoder = self.encoder(ten)
            # decode the tensor
            ten = self.decoder(ten_encoder)
            # decoder for samples
            return ten, ten_encoder
        else:
            if not only_decode:
                encoding = self.encoder(ten)
                reconstruct = self.decoder(encoding)
                return reconstruct, encoding
            else:
                return self.decoder(ten)

    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original,
             labels_sampled):
        """

        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :return:
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        L2Loss = nn.MSELoss()

        nle = L2Loss(ten_original, ten_predict)
        # kl-divergence
        # mse between intermediate layers
        mse = L2Loss(layer_original, layer_predicted)
        # bce for decoder and discriminator for original,sampled and reconstructed
        # the only excluded is the bce_gen_original

        bce_dis_original = -torch.log(labels_original + 1e-3)
        bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)

        bce_gen_original = -torch.log(1 - labels_original + 1e-3)
        bce_gen_sampled = -torch.log(labels_sampled + 1e-3)

        return nle, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sampled
