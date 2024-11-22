import torch
import random
from typing import Callable
from src.diffusion.base.training import *

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1

class DEQTrainer(BaseTrainer):
    def __init__(
            self,
            loss_weight_fn:Callable=constant,
            max_buffer_length=4096,
            buffer_batch_size=16,
            noise_p=1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_weight_fn = loss_weight_fn
        self.data_buffer = []
        self.max_buffer_length = max_buffer_length
        self.buffer_batch_size = buffer_batch_size
        self.noise_p = noise_p

    def _add_sample(self, images, noise, labels, v_t):
        batch_size = images.shape[0]
        p = torch.rand(batch_size).to(images.device, images.dtype)
        p = p.view(batch_size, 1, 1, 1) * self.noise_p
        updated_noise = (noise + v_t) + p*torch.randn_like(noise)
        noise_list = updated_noise.unbind(0)
        image_list = images.unbind(0)
        label_list = labels.unbind(0)
        pair_list = list(zip(noise_list, image_list, label_list))
        self.data_buffer.extend(pair_list)
        while len(self.data_buffer) > self.max_buffer_length:
            self.data_buffer.pop(0)

    def _get_buffer_sample(self):
        buffer_data = random.sample(self.data_buffer, self.buffer_batch_size)
        noise_list, image_list, label_list = zip(*buffer_data)
        images = torch.stack(image_list)
        noise = torch.stack(noise_list)
        labels = torch.stack(label_list)
        return images, noise, labels


    def _impl_trainstep(self, net, images, labels):
        noise = torch.randn_like(images)
        if len(self.data_buffer) > self.buffer_batch_size:
            buffer_images, buffer_noise, buffer_labels = self._get_buffer_sample()
            images = torch.cat([images, buffer_images], dim=0)
            noise = torch.cat([noise, buffer_noise], dim=0)
            labels = torch.cat([labels, buffer_labels], dim=0)
        batch_size = images.shape[0]
        t = torch.zeros(batch_size).to(images.device, images.dtype)
        v_t = images - noise
        out = net(noise, t, labels)
        loss = (out - v_t)**2
        self._add_sample(images, noise, labels, out.detach())
        out = dict(
            loss=loss.mean(),
        )
        return out