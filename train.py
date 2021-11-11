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

from model.discriminator import Discriminator
from model.generator import Generator
from model.feature_extractor import FeatureExtractor


target_size = 256

anime_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

smooth_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

photo_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
photo_image_dir = "/content/train_photo/"
animation_image_dir = "/content/Shinkai/style/"
edge_smoothed_dir = "/content/Shinkai/smooth/"
test_photo__dir = "/content/val/"
ckpt_path = "/content/checkpoints/"



adam_beta1 = 0.5  # following dcgan
lr = 0.0002
num_epochs = 100
initialization_epochs = 10
content_loss_weight = 10
print_every = 100

class CartoonGAN():

    def __init__(self, generator, discriminator, feature_extractor, dataloader, content_loss_weight):

        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.dataloader = dataloader
        self.content_loss_weight = content_loss_weight

        self.generator_optim = optim.Adam(self.generator.parameters(), lr=lr, betas=(adam_beta1, 0.999))
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(adam_beta1, 0.999))

        self.discriminator_criterion = nn.MSELoss()
        self.generator_criterion = nn.MSELoss()
        self.content_criterion = nn.L1Loss()

        self.print_every = print_every
        self.base_img = photo_transforms(io.imread("/content/result/download (4).jpeg")).unsqueeze(0).to(device)

    def train_step(self, anime_img, edge_smoothed_img, photo_img):

        self.discriminator.zero_grad()
        self.generator.zero_grad()

        loss_D = 0
        loss_G = 0
        loss_content = 0

        # 1. Train Discriminator
        # 1-1. Train Discriminator using animation images
        animation_disc_output = self.discriminator(anime_img)
        animation_target = torch.ones_like(animation_disc_output)
        loss_real = self.discriminator_criterion(animation_disc_output, animation_target)

        # 1-2. Train Discriminator using edge smoothed images
        edge_smoothed_disc_output = self.discriminator(edge_smoothed_img)
        edge_smoothed_target = torch.zeros_like(edge_smoothed_disc_output)
        loss_edge = self.discriminator_criterion(edge_smoothed_disc_output, edge_smoothed_target)

        # 1-3. Train Discriminator using generated images
        generated_images = self.generator(photo_img).detach()

        generated_output = self.discriminator(generated_images)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.discriminator_criterion(generated_output, generated_target)

        loss_disc = loss_real + loss_edge + loss_generated

        loss_disc.backward()
        loss_D = loss_disc.item()

        self.discriminator_optim.step()

        # 2. Train Generator
        self.generator.zero_grad()

        # 2-1. Train Generator using adversarial loss, using generated images
        generated_images = self.generator(photo_img)
        base_images = self.generator(self.base_img)

        generated_output = self.discriminator(generated_images)
        generated_target = torch.ones_like(generated_output)
        loss_adv = self.generator_criterion(generated_output, generated_target)

        # 2-2. Train Generator using content loss
        x_features = self.feature_extractor((photo_img + 1) / 2).detach()
        Gx_features = self.feature_extractor((generated_images + 1) / 2)

        loss_content = self.content_loss_weight * self.content_criterion(Gx_features, x_features)

        loss_gen = loss_adv + loss_content
        loss_gen.backward()

        loss_G = loss_adv.item()
        loss_content = loss_content.item()

        self.generator_optim.step()

        return loss_D, loss_G, loss_content, generated_images, base_images

        


    def train(self, num_epochs=num_epochs):

        wandb.init(project="Photo2Anime", name="session-1")

        for epoch in range(num_epochs):

            epoch_loss_D = 0
            epoch_loss_G = 0
            epoch_loss_content = 0

            for i, (anime_imgs, smooth_imgs, photo_imgs) in enumerate(self.dataloader):

                anime_imgs = anime_imgs.to(device)
                smooth_imgs = smooth_imgs.to(device)
                photo_imgs = photo_imgs.to(device)
                

                loss_D, loss_G, loss_content, generated_images, base_images = self.train_step(anime_imgs, smooth_imgs, photo_imgs)

                epoch_loss_D += loss_D
                epoch_loss_G += loss_G
                epoch_loss_content += loss_content

                if ((i + 1) % print_every) == 0:

                    print(f"[Epoch - {epoch + 1}/{num_epochs}] [Step - {i + 1}/{len(dataloader)}] [Average Generator Loss - {epoch_loss_G/(i+1)}] [Average Discriminator Loss - {epoch_loss_D/(i+1)}] [Average Content Loss - {epoch_loss_content/(i+1)}]")

                    wandb.log({"Anime": [wandb.Image((255 * np.array(generated_images[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))],
                               "Photo": [wandb.Image((255 * np.array(photo_imgs[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))],
                               "Base": [wandb.Image((255 * np.array(base_images[0].transpose(0, 1).transpose(1, 2).cpu().detach() * 2) + 0.5))]
                    })

                    

            torch.save(self.generator.state_dict(), f"epoch_{epoch}_generator.pth")

            print("=> Model Saved")


generator = Generator().to(device)
discriminator = Discriminator().to(device)
feature_extractor = FeatureExtractor().to(device)
dataset = PreMadeDataset(animation_image_dir, edge_smoothed_dir, photo_image_dir,
                                 anime_transform=anime_transforms,
                                 smooth_transform=smooth_transforms,
                                 photo_transform=photo_transforms)
    
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
gan = CartoonGAN(generator, discriminator, feature_extractor, dataloader,
                        content_loss_weight=content_loss_weight)

gan.train()
