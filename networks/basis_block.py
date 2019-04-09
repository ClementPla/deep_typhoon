import torch.nn.functional as F
import torch.nn as nn


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, upsampling='tranposed'):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        if upsampling == 'transposed':
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2,
                                           output_padding=1,
                                           bias=False)
        else:
            self.conv = nn.Sequential(*[nn.Upsample(scale_factor=2, mode='nearest'),
                                       nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5,
                                                 padding=2, stride=1,
                                                 bias=False)
                                       ])
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Decoder(nn.Module):
    def __init__(self, z_size, size, upsampling='transposed'):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                nn.ReLU(True))

        layers_list = []
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling))
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling))
        for i in range(4):
            layers_list.append(DecoderBlock(channel_in=int(size * 2 ** (-i)), channel_out=int(size * 2 ** (-i - 1)),
                                            upsampling=upsampling))

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
