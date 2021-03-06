import torch.nn.functional as F
import torch.nn as nn


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, norm='batch', activation=nn.LeakyReLU(0.2, True)):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.norm = norm
        if norm == 'batch':
            self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                                  bias=False)
            self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        elif norm == 'none' or norm is None:
            self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                                  bias=True)
        elif norm == 'instance':
            self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                                  bias=False)
            self.bn = nn.InstanceNorm2d(channel_out, True)

        self.activation = activation

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if self.norm in ['batch', 'instance']:
            if out:
                ten = self.conv(ten)
                ten_out = ten
                ten = self.bn(ten)
                ten = self.activation(ten)
                return ten, ten_out
            else:
                ten = self.conv(ten)
                ten = self.bn(ten)
                ten = self.activation(ten)
                return ten
        elif self.norm == 'none' or self.norm is None:
            if out:
                ten = self.conv(ten)
                return self.activation(ten), ten
            else:
                ten = self.conv(ten)
                return self.activation(ten)

# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, upsampling='transposed', norm='batch', activation=nn.LeakyReLU(0.2, True)):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.channel_out = channel_out
        self.norm = norm
        if norm in ['batch', 'instance']:
            if upsampling == 'transposed':
                self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2,
                                               output_padding=1,
                                               bias=False)
            else:
                self.conv = nn.Sequential(*[nn.Upsample(scale_factor=2, mode=upsampling),
                                           nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5,
                                                     padding=2, stride=1,
                                                     bias=False)
                                           ])
            if norm == 'batch':
                self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
            elif norm == 'instance':
                self.bn = nn.InstanceNorm2d(channel_out, True)
        elif norm == 'none' or norm is None:
            if upsampling == 'transposed':
                self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2,
                                               output_padding=1,
                                               bias=True)
            else:
                self.conv = nn.Sequential(*[nn.Upsample(scale_factor=2, mode=upsampling),
                                           nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5,
                                                     padding=2, stride=1,
                                                     bias=True)
                                           ])
        self.activation = activation

    def forward(self, ten):
        ten = self.conv(ten)
        if self.norm in ['batch', 'instance']:
            ten = self.bn(ten)
        ten = self.activation(ten)
        return ten


class Decoder(nn.Module):
    def __init__(self, z_size, size, upsampling='transposed', get_single_levels=False, activation=nn.ReLU(True)):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
                                activation)

        self.get_single_levels = get_single_levels
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling, activation=activation))
        layers_list.append(DecoderBlock(channel_in=size, channel_out=size, upsampling=upsampling, activation=activation))
        for i in range(4):
            layers_list.append(DecoderBlock(channel_in=int(size * 2 ** (-i)), channel_out=int(size * 2 ** (-i - 1)),
                                            upsampling=upsampling, activation=activation))

        self.size = int(size * 2 ** (-i - 1))
        # final conv to get 1 channels and tanh layer
        if not self.get_single_levels:
            layers_list.append(nn.Sequential(
                nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.Tanh()
            ))

            self.conv = nn.Sequential(*layers_list)
        else:
            self.progressive_convs = layers_list
            self.outputs_convs = []
            for i, layer in enumerate(self.progressive_convs):
                setattr(self, 'upconv_%i'%i, layer)
                out = nn.Sequential(nn.Conv2d(in_channels=layer.channel_out, out_channels=1,
                                                                  kernel_size=5, stride=1, padding=2), nn.Tanh())

                setattr(self, 'outconv_%i'%i, out)
                self.outputs_convs.append(out)

    def forward(self, ten, only_last=False):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 8, 8)
        if not self.get_single_levels:
            ten = self.conv(ten)
            return ten
        else:
            outs = []
            for layer, output in zip(self.progressive_convs, self.outputs_convs):
                ten = layer(ten)
                outs.append(output(ten))
            if only_last:
                return outs[-1]
            else:
                return outs