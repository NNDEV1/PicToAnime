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

import model_utils

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        block1 = [nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, stride=1),
                       nn.LeakyReLU(0.2, inplace=True)]

        block2 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
                       nn.LeakyReLU(0.2, inplace=True),
                       nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
                       nn.BatchNorm2d(128),
                       nn.LeakyReLU(0.2, inplace=True)]

        block3 = [nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2, padding=1),
                       nn.LeakyReLU(0.2, inplace=True),
                       nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
                       nn.BatchNorm2d(256),
                       nn.LeakyReLU(0.2, inplace=True)]

        block4 = [nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
                       nn.BatchNorm2d(256),
                       nn.LeakyReLU(0.2, inplace=True)]

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        out_block = [nn.Conv2d(256, 1, kernel_size=(3, 3), stride=1, padding=1),
                     nn.Sigmoid()]

        self.out_block = nn.Sequential(*out_block)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        out = self.out_block(x)

        return out
