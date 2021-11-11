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

class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x + residual
      
      
class UpSample2D(nn.Module):

    def __init__(self, in_channels):
        super(UpSample2D, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
      
class DownSample2D(nn.Module):

    def __init__(self, in_channels):
        super(DownSample2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, stride=2, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, stride=1, kernel_size=(3, 3), padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
      
