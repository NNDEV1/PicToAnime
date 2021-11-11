from PIL import Image
import PIL
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from skimage import io
from torchsummary import summary
import time
import torchvision.utils as vutils
import torchvision
from torch import optim
import glob
import wandb
import random 

class Generator(nn.Module):

    def __init__(self, n_res_blocks=8):
        super(Generator, self).__init__()

        self.in_block = [nn.Conv2d(3, 64, (7, 7), stride=1, padding=3, bias=False),
                         nn.InstanceNorm2d(64),
                         nn.LeakyReLU(inplace=True)]

        self.in_block = nn.Sequential(*self.in_block)

        self.downsample1 = DownSample2D(64)
        self.downsample2 = DownSample2D(64)

        self.res_blocks = []

        for i in range(n_res_blocks):

            self.res_blocks.append(ResidualBlock(64))

        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.upsample1 = UpSample2D(64)
        self.upsample2 = UpSample2D(64)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.in_block(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.out_conv(x)
        x = self.tanh(x)

        return x
