import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import torch


def reverse_z(netG, goal, z_size=None, cuda=None, clip='disabled', lr=0.001, niter=500, initial_z=None, loss=nn.MSELoss, **kwargs):
    # sanity check
    assert clip in ['disabled', 'standard', 'stochastic']
    print("Optimizing z with %i iterations"%niter)

    # loss metrics
    mse_loss = loss()
    if initial_z is None:
        z_approx = torch.randn(goal.size()[0], z_size).cuda(cuda)
    else:
        z_approx = initial_z

    # transfer to gpu
    mse_loss.cuda(cuda)

    # convert to variable
    z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    # optimizer
    optimizer_approx = optim.Adam([z_approx], lr=lr)

    # train
    for i in range(niter):
        g_z_approx = netG(z_approx, only_last=True)
        mse_g_z = mse_loss(g_z_approx, goal)

        # bprop
        optimizer_approx.zero_grad()
        mse_g_z.backward()
        optimizer_approx.step()

        # clipping
        if clip == 'standard':
            z_approx.data[z_approx.data > 1] = 1
            z_approx.data[z_approx.data < -1] = -1
        if clip == 'stochastic':
            z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
            z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)

    return z_approx


def get_angle(z1, z2):
    z1 = z1.flatten()
    z2 = z2.flatten()
    theta = torch.acos(torch.dot(z1, z2) / (z1.norm() * z2.norm()))
    return theta