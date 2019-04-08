import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
from .abstract_network import AbstractNet
from .basis_block import *


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
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


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


class VaeGan(AbstractNet):
    def __init__(self, z_size=128, recon_level=3, gpu=1):
        super(VaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=1, recon_level=recon_level)
        # self-defined function to init the parameters
        self.init_parameters()
        self.gpu = gpu

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

    def forward(self, ten, gen_size=10):
        if self.training:
            # save the original images
            ten_original = ten
            # encode
            mus, log_variances = self.encoder(ten)
            # we need the true variances, not the log one
            variances = torch.exp(log_variances * 0.5)
            # sample from a gaussian

            if self.gpu is not None:
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(self.gpu), requires_grad=True)
            else:
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size), requires_grad=True)
                # shift and scale using the means and variances

            ten = ten_from_normal * variances + mus
            # decode the tensor
            ten = self.decoder(ten)
            # discriminator for reconstruction
            ten_layer = self.discriminator(ten, ten_original, "REC")
            # decoder for samples
            if self.gpu is not None:
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(self.gpu), requires_grad=True)
            else:
                ten_from_normal = Variable(torch.randn(len(ten), self.z_size), requires_grad=True)

            ten = self.decoder(ten_from_normal)
            ten_class = self.discriminator(ten_original, ten, "GAN")
            return ten, ten_class, ten_layer, mus, log_variances
        else:
            if ten is None:
                # just sample and decode
                if self.gpu is not None:
                    ten = Variable(torch.randn(gen_size, self.z_size).cuda(self.gpu), requires_grad=False)
                else:
                    ten = Variable(torch.randn(gen_size, self.z_size), requires_grad=False)

                ten = self.decoder(ten)
            else:
                mus, log_variances = self.encoder(ten)
                # we need the true variances, not the log one
                variances = torch.exp(log_variances * 0.5)
                # sample from a gaussian
                if self.gpu is not None:
                    ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(self.gpu), requires_grad=False)
                else:
                    ten_from_normal = Variable(torch.randn(len(ten), self.z_size), requires_grad=False)
                    # shift and scale using the means and variances
                ten = ten_from_normal * variances + mus
                # decode the tensor
                ten = self.decoder(ten)
            return ten

    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original,
             labels_sampled, mus, variances):
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
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
        # mse between intermediate layers
        mse = L2Loss(layer_original, layer_predicted)
        # bce for decoder and discriminator for original,sampled and reconstructed
        # the only excluded is the bce_gen_original

        bce_dis_original = -torch.log(labels_original + 1e-3)
        bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)

        bce_gen_original = -torch.log(1 - labels_original + 1e-3)
        bce_gen_sampled = -torch.log(labels_sampled + 1e-3)
        '''


        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''
        return nle, kl, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sampled
