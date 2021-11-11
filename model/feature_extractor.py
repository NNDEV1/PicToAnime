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

class FeatureExtractor(nn.Module):

    def __init__(self, network="resnet-101"):
        super(FeatureExtractor, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=True)
        layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2]
        self.feature_extractor = nn.Sequential(*layers)

        for child in self.feature_extractor.children():
            for param in child.parameters():

                param.requires_grad = False

    def forward(self, input):

        return self.feature_extractor(input)
