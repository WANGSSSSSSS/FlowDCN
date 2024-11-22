import torch
from torch import Tensor
from typing import Callable

class BaseSampler:
    def __init__(self,
                 null_class,
                 guidance_fn: Callable,
                 num_steps: int = 250,
                 guidance: float = 1.0,
                 *args,
                 **kwargs
        ):
        self.null_class = null_class
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn

    def preprocess(self, images, labels):
        return images, labels

    def postprocess(self, images):
        return images
    def _impl_sampling(self, net, images, labels):
        raise NotImplementedError

    def __call__(self, net, images, labels):
        images, labels = self.preprocess(images, labels)
        denoised = self._impl_sampling(net, images, labels)
        denoised = self.postprocess(denoised)
        return denoised


