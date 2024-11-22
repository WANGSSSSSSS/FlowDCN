import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

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

class FlowMatchingTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            lognorm_t=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
    def _impl_trainstep(self, net, images, labels):
        batch_size = images.shape[0]
        if self.lognorm_t:
            t = torch.randn(batch_size).to(images.device, images.dtype).sigmoid()
        else:
            t = torch.rand(batch_size).to(images.device, images.dtype)
        noise = torch.randn_like(images)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)
        w = self.scheduler.w(t)

        x_t = alpha * images + noise * sigma
        v_t = dalpha * images + dsigma * noise
        out = net(x_t, t, labels)
        weight = self.loss_weight_fn(alpha, sigma)

        loss = weight*(out - v_t)**2

        out = dict(
            loss=loss.mean(),
        )
        return out