import torch

from src.diffusion.base.guidance import *
from src.diffusion.base.sampling import *

from typing import Callable


import logging
logger = logging.getLogger(__name__)

class DEQSampler(BaseSampler):
    def __init__(
            self,
            guidance_fn: Callable = c3_guidance_fn,
            noise_p=0.2,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.guidance_fn = guidance_fn
        self.noise_p = noise_p

    def _impl_sampling(self, net, images, labels):
        """
        sampling process of Euler sampler
        -
        """
        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        t = torch.zeros(images.shape[0]).to(images.device, images.dtype)
        for i  in range(self.num_steps):
            cfg_x = torch.cat([x, x], dim=0)
            t_cur = t.repeat(2)
            out = net(cfg_x, t_cur, labels)
            out = self.guidance_fn(out, self.guidance)
            x = x + out + torch.randn_like(x) * self.noise_p
        return x