from diffusers import AutoencoderKL

import torch
from typing import Callable
from torchvision.datasets import ImageFolder, ImageNet

import cv2
import os
import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torchmetrics.image import FrechetInceptionDistance

from PIL import Image
import pathlib

import torch
import random
from torchvision.io.image import read_image
import torchvision.transforms as tvtf
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

IMG_EXTENSIONS = (
    "*.png",
    "*.JPEG",
    "*.jpeg",
    "*.jpg"
)
import time
class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        def center_crop_arr(pil_image, image_size):
            """
            Center cropping implementation from ADM.
            https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
            """
            while min(*pil_image.size) >= 2 * image_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image)
            crop_y = (arr.shape[0] - image_size) // 2
            crop_x = (arr.shape[1] - image_size) // 2
            return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

        return center_crop_arr(image, self.size)



if __name__ == "__main__":

    for split in ['train']:
        train = split == 'train'
        transforms = tvtf.Compose([
            CenterCrop(256),
            # tvtf.RandomHorizontalFlip(p=1),
            tvtf.ToTensor(),
            tvtf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = ImageFolder(root='data/imagenet/train', transform=transforms)
        B = 24
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, prefetch_factor=4, num_workers=2,)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")#.to('cuda')

        from accelerate import Accelerator

        accelerator = Accelerator()

        vae, dataloader = accelerator.prepare( vae, dataloader)
        rank = accelerator.process_index
        base = rank * (len(dataloader) + 1)
        print(base)
        with torch.no_grad():
            for i, (image, label) in enumerate(dataloader):
                 image = image.to("cuda")
                 distribution = vae.module.encode(image).latent_dist
                 mean = distribution.mean
                 logvar = distribution.logvar

                 # horizontal flip
                 fliped_image = image.flip(dims=[3])
                 distribution = vae.module.encode(fliped_image).latent_dist
                 fliped_mean = distribution.mean
                 fliped_logvar = distribution.logvar

                 for j in range(B):
                     out = dict(
                         mean=mean[j].cpu(),
                         logvar=logvar[j].cpu(),
                         label=label[j].cpu(),
                     )
                     out.update(
                         fliped_mean=fliped_mean[j].cpu(),
                         fliped_logvar=fliped_logvar[j].cpu(),
                     )
                     indx = base*B + i*B + j
                     torch.save(out, f'data/imagenet_latent_fliped/{indx}.pt')

                 # decoded_image = vae.decode(latent).sample
                 # decoded_image = decoded_image.clamp(-1, 1)
                 # to image and save
                 # decoded_image = (decoded_image + 1) / 2
                 # decoded_image = decoded_image.permute(0, 2, 3, 1)
                 # decoded_image = decoded_image.cpu().numpy()
                 # decoded_image = decoded_image * 255
                 # decoded_image = decoded_image.astype(np.uint8)
                 # decoded_image = decoded_image.reshape(-1, 256, 256, 3)
                 # for j in range(decoded_image.shape[0]):
                 #     image = decoded_image[j]
                 #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                 #     cv2.imwrite(f'{i}_{j}.png', image)
                 #     exit()