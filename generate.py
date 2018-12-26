from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json

import models.dcgan as dcgan
import models.mlp as mlp

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--nimages', required=True, type=int, help="number of images to generate", default=1)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()

    with open(opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())
    
    imageSize = generator_config["imageSize"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    noBN = generator_config["noBN"]
    ngpu = generator_config["ngpu"]
    mlp_G = generator_config["mlp_G"]
    n_extra_layers = generator_config["n_extra_layers"]

    if noBN:
        netG = dcgan.DCGAN_G_nobn(imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif mlp_G:
        netG = mlp.MLP_G(imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # load weights
    netG.load_state_dict(torch.load(opt.weights))

    # initialize noise
    fixed_noise = torch.FloatTensor(opt.nimages, nz, 1, 1).normal_(0, 1)

    if opt.cuda:
        netG.cuda()
        fixed_noise = fixed_noise.cuda()

    fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)

    for i in range(opt.nimages):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(opt.output_dir, "generated_%02d.png"%i))
