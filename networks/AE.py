import torch.nn as nn
import numpy
from .abstract_network import AbstractNet
from .basis_block import *
from utils.tensors import remove_nan

class Encoder(nn.Module):
    def __init__(self, channel_in=1, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        layers_list += [nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, padding=2, stride=1,
                              bias=False), nn.BatchNorm2d(num_features=32, momentum=0.9)]
        for i in range(4):
            layers_list.append(EncoderBlock(channel_in=32*(2**i), channel_out=32*2**(i+1)))

        self.size = 32*2**(i+1)

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


class Decoder(nn.Module):
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(True))

        layers_list = []
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size))
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size))
        for i in range(4):
            layers_list.append(DecoderBlock(channel_in=int(size * 2 ** (-i)), channel_out=int(size * 2 ** (-i - 1))))

        self.size = int(size * 2 ** (-i - 1))
        # final conv to get 1 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        ten = self.conv(ten)
        return ten


class AE(AbstractNet):
    def __init__(self, channel_in=1, z_size=128, gpu=1, checkpoint=''):
        super(AE, self).__init__(gpu=gpu, checkpoint=checkpoint)
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size, channel_in=channel_in)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        # self-defined function to init the parameters
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
            # save the original images
            ten_original = ten
            # encode
            ten_encoder = self.encoder(ten)
            # decode the tensor
            ten = self.decoder(ten_encoder)
            # discriminator for reconstruction
            return ten
        else:
            if not only_decode:
                encoding = self.encoder(ten)
                reconstruct = self.decoder(encoding)
                return reconstruct, encoding
            else:
                return self.decoder(ten)


    @staticmethod
    def loss(ten_original, ten_predict):
        """

        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        L2Loss = nn.MSELoss()
        remove_nan(ten_original, ten_predict)
        nle = L2Loss(ten_original, ten_predict)
        return nle
