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

    def forward(self, ten, get_last_conv=False):
        conv_out = self.conv(ten)
        ten = conv_out.view(len(conv_out), -1)
        ten = self.fc(ten)
        if get_last_conv:
            return ten
        else:
            return ten, conv_out


class AE(AbstractNet):
    def __init__(self, channel_in=1, z_size=128, gpu=1, checkpoint='', upsampling='transposed'):
        super(AE, self).__init__(gpu=gpu, checkpoint=checkpoint, upsampling=upsampling)
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

    def forward(self, ten, only_decode=False, get_last_encoder_conv=False):
        if self.training:
            if not get_last_encoder_conv:
                # save the original images
                # encode
                ten_encoder = self.encoder(ten)
                # decode the tensor
                ten = self.decoder(ten_encoder)
                # discriminator for reconstruction
                return ten
            else:
                ten_encoder, ten_conv_encoder = self.encoder(ten, get_last_encoder_conv)
                ten = self.decoder(ten_encoder)
                return ten, ten_conv_encoder
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

        L2Loss = nn.MSELoss()
        nle = L2Loss(ten_original, ten_predict)
        return nle

    def rec_loss(self, ten_original, ten_predict, encoder_conv):
        """

        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        """

        L2Loss = nn.MSELoss()
        nle = L2Loss(ten_original, ten_predict)

        _, rec_encoder_conv = self.encoder(ten_predict, True)
        rec_loss = L2Loss(rec_encoder_conv, encoder_conv)
        return nle, rec_loss
