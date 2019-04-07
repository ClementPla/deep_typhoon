import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy
from .abstract_network import AbstractNet


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
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    def __init__(self, channel_in=1, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(6):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=32))
                self.size = 32
            elif i == 4:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size))
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
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
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))
        layers_list.append(DecoderBlock(channel_in=self.size // 2, channel_out=self.size // 4))
        layers_list.append(DecoderBlock(channel_in=self.size // 4, channel_out=self.size // 8))
        layers_list.append(DecoderBlock(channel_in=self.size // 8, channel_out=self.size // 16))
        layers_list.append(DecoderBlock(channel_in=self.size // 16, channel_out=self.size // 16))
        layers_list.append(DecoderBlock(channel_in=self.size // 16, channel_out=self.size // 32))
        self.size = self.size // 32
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
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return torch.sigmoid(ten)


class CAEGan(AbstractNet):
    def __init__(self, channel_in=1, z_size=128, recon_level=3, gpu=1, checkpoint=''):
        super(CAEGan, self).__init__(gpu=gpu, checkpoint=checkpoint)
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size, channel_in=channel_in)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=channel_in, recon_level=recon_level)
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

    def forward(self, ten, decode=False):
        if self.training:
            # save the original images
            ten_original = ten
            # encode
            ten_encoder = self.encoder(ten)
            # decode the tensor
            ten = self.decoder(ten_encoder)
            # discriminator for reconstruction
            ten_layer = self.discriminator(ten, ten_original, "REC")
            # decoder for samples
            ten_class = self.discriminator(ten_original, ten, "GAN")
            return ten, ten_class, ten_layer, ten_encoder
        else:
            if decode:
                encoding = self.encoder(ten)
                reconstruct = self.decoder(encoding)
                return reconstruct, encoding
            else:
                return self.decoder(ten)


    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original,
             labels_sampled, h):
        """

        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
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

        dh = h * (1 - h)  # Hadamard product produces size N_batch x N_hidden
        # Sum through the input dimension to improve efficiency, as suggested in #1
        w_sum = torch.sum(Variable(W) ** 2, dim=1)
        # unsqueeze to avoid issues with torch.mv
        w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1

        contractive_loss = torch.sum(torch.mm(dh ** 2, w_sum), 0)

        return nle, contractive_loss, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sampled
