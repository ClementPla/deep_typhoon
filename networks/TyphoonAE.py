import torch.nn as nn
import numpy
from .abstract_network import AbstractNet
from .basis_block import *
import torch

class TyphoonEncoder(nn.Module):
    def __init__(self, channel_in=1, z_size=128):
        super(TyphoonEncoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 1->32, for every other double the channel size
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
                                nn.Tanh())

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        return ten


class TyphoonDecoder(nn.Module):
    def __init__(self, z_size, size, upsampling='transposed', get_single_levels=False):
        super(TyphoonDecoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(True),
                                nn.Linear(in_features=1024, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(True)
                                )

        self.multiscale = get_single_levels
        self.layers_list = nn.ModuleList()
        self.layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling))
        self.layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling))
        for i in range(4):
            self.layers_list.append(DecoderBlock(channel_in=int(size * 2 ** (-i)), channel_out=int(size * 2 ** (-i - 1)),
                                            upsampling=upsampling))

        self.size = int(size * 2 ** (-i - 1))
        # final conv to get 1 channels and tanh layer
        if not self.multiscale:
            self.layers_list.append(nn.Sequential(
                nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.Tanh()
            ))

            self.conv = nn.Sequential(*self.layers_list)
        else:
            self.outputs_convs = nn.ModuleList()
            for i, layer in enumerate(self.layers_list):
                out = nn.Sequential(nn.Conv2d(in_channels=layer.channel_out, out_channels=1,
                                                                  kernel_size=5, stride=1, padding=2), nn.Tanh())

                self.outputs_convs.append(out)

    def forward(self, ten, only_last=False):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        if not self.multiscale:
            ten = self.conv(ten)
            return ten
        else:
            outs = []
            for layer, output in zip(self.layers_list, self.outputs_convs):
                ten = layer(ten)
                outs.append(output(ten))
            if only_last:
                return outs[-1]
            else:
                return outs


class Discriminator(nn.Module):
    def __init__(self, channel_in=1, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=64))
        self.size = 64
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=512))
        self.size = 512
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=512))

        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 512, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, ten, other_ten, mode='REC'):
        if mode == "REC":
            b = ten.size(0)
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten[:b], layer_ten[b:]
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return torch.sigmoid(ten)


class TyphoonAE(AbstractNet):
    def __init__(self, channel_in=1,
                 z_size=128,
                 rec_level=3,
                 gpu=1,
                 checkpoint='',
                 upsampling='nearest',
                 gan=False):

        super(TyphoonAE, self).__init__(gpu=gpu, checkpoint=checkpoint, upsampling=upsampling)

        self.z_size = z_size
        self.encoder = TyphoonEncoder(z_size=self.z_size, channel_in=channel_in)
        self.decoder = TyphoonDecoder(z_size=self.z_size, size=self.encoder.size, upsampling=upsampling,
                                      get_single_levels=True)

        if gan:
            self.discriminator = Discriminator(channel_in, recon_level=rec_level)

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    nn.init.xavier_normal_(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    # nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, ten, only_decode=False, only_last_scale=False):
        encoding = self.encoder(ten)
        reconstruct = self.decoder(encoding)
        if not only_decode:
            if only_last_scale:
                return reconstruct[-1], encoding
            else:
                return reconstruct, encoding
        else:
            if only_last_scale:
                return reconstruct[-1]
            else:
                return reconstruct


