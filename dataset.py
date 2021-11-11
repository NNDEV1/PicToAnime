# to get data use
# wget https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip

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

files = os.listdir("/content/train_photo")  # Get filenames in current folder
files = random.sample(files, 5006)  # Pick 5006 random files
for file in files:  # Go over each file name to be deleted
    f = os.path.join("/content/train_photo", file)  # Create valid path to file
    os.remove(f)  # Remove the file
    
def make_edges_smooth(img, kernel_size=9, canny_threshold1=30, canny_threshold2=60):

    img = (np.array(img).astype(np.uint8))
    edges = cv2.Canny(img, canny_threshold1, canny_threshold2)

    dilated_edges = cv2.dilate(edges, (7, 7), iterations=25)
    dilated_edges_to_compare = dilated_edges.copy()
    dilated_edges_to_compare[dilated_edges == 0] = -1

    img_no_dilated_edges, img_only_dilated_edges = img.copy(), img.copy()
    img_no_dilated_edges[dilated_edges_to_compare != -1] = 0
    img_only_dilated_edges[dilated_edges_to_compare == -1] = 0

    blurred_edges = cv2.GaussianBlur(img_only_dilated_edges, (kernel_size, kernel_size), 0)

    blurred_edges[dilated_edges_to_compare == -1] = 0

    result = blurred_edges + img_no_dilated_edges

    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

    return result
class CustomAnime2PhotoDataset(Dataset):

    def __init__(self, anime_root_path, photo_root_path, anime_transform_a=None, anime_transform_b=None, photo_transform=None):

        self.anime_root_path = anime_root_path
        self.photo_root_path = photo_root_path
        self.anime_transform_a = anime_transform_a
        self.anime_transform_b = anime_transform_b
        self.photo_transform = photo_transform
        self.anime_files = os.listdir(anime_root_path)
        self.photo_files = os.listdir(photo_root_path)

    def __len__(self):

        return min(len(self.anime_files), len(self.photo_files))

    def __getitem__(self, index):

        anime_img_path = os.path.join(self.anime_root_path, self.anime_files[index])
        photo_img_path = os.path.join(self.photo_root_path, self.photo_files[index])

        self.anime_img = io.imread(anime_img_path)
        self.photo_img = io.imread(photo_img_path)

        if self.anime_transform_a:

            self.anime_img = self.anime_transform_a(self.anime_img)

        if self.photo_transform:

            self.photo_img = self.photo_transform(self.photo_img)

        self.smooth_anime_img = make_edges_smooth(self.anime_img)

        if self.anime_transform_b:

            self.anime_img = self.anime_transform_b(self.anime_img)
            self.smooth_anime_img = self.anime_transform_b(self.smooth_anime_img)

        return (self.anime_img, self.smooth_anime_img, self.photo_img)


class PreMadeDataset(Dataset):

    def __init__(self, anime_dir, smooth_dir, photo_dir, anime_transform=None, smooth_transform=None, photo_transform=None):

        self.anime_dir = anime_dir
        self.smooth_dir = smooth_dir
        self.photo_dir = photo_dir
        self.anime_transform = anime_transform
        self.smooth_transform = smooth_transform
        self.photo_transform = photo_transform
        self.anime_files = os.listdir(anime_dir)
        self.photo_files = os.listdir(photo_dir)
        self.smooth_files = os.listdir(smooth_dir)

    def __len__(self):

        return len(self.photo_files)

    def __getitem__(self, index):

        anime_img_path = os.path.join(self.anime_dir, self.anime_files[index])
        photo_img_path = os.path.join(self.photo_dir, self.photo_files[index])
        smooth_img_path = os.path.join(self.smooth_dir, self.smooth_files[index])

        self.anime_img = io.imread(anime_img_path)
        self.photo_img = io.imread(photo_img_path)
        self.smooth_img = io.imread(smooth_img_path)

        if self.photo_transform:

            self.photo_img = self.photo_transform(self.photo_img)

        if self.anime_transform:

            self.anime_img = self.anime_transform(self.anime_img)

        if self.smooth_transform:

            self.smooth_img = self.smooth_transform(self.smooth_img)

        return (self.anime_img, self.smooth_img, self.photo_img)
