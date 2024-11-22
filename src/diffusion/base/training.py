import time

import torch
import torch.nn as nn

class BaseTrainer:
    def __init__(self,
                 null_class=1000,
                 null_condition_p=0.1,
                 log_var=False,
        ):
        self.null_class = null_class
        self.null_condition_p = null_condition_p
        self.log_var = log_var
    def setup(self, vae):
        self.vae = vae
    def preproprocess(self, images, labels):
        if labels is None:
            labels = torch.full(images.shape[0], self.null_class, dtype=torch.long, device=images.device)

        if self.null_condition_p > 0:
            mask = torch.rand(labels.shape, device=labels.device) < self.null_condition_p
            labels[mask] = self.null_class
        return images, labels

    def _impl_trainstep(self, net, images, labels):
        raise NotImplementedError

    def __call__(self, net, images, labels):
        images, labels = self.preproprocess(images, labels)
        images = self.vae.encode(images)
        return self._impl_trainstep(net, images, labels)

