import torch

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable


def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt

def sde_step_fn(x, v, dt, s, w):
    return x + v*dt + s * w* dt + torch.sqrt(2*w*dt)*torch.randn_like(x)

def sde_preserve_step_fn(x, v, dt, s, w):
    return x + v*dt + 0.5*s*w* dt + torch.sqrt(w*dt)*torch.randn_like(x)


class ABSFlowMatchEulerSampler(BaseSampler):
    def __init__(
            self,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            last_step=0.0,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        if self.last_step == 0.0:
            self.last_step = 1.0 / self.num_steps
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]

    def _impl_sampling(self, net, images, labels):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = images.shape[0]
        steps = torch.linspace(0.0, 1 - self.last_step, self.num_steps, device=images.device)
        steps = torch.cat([steps, torch.tensor([1.0], device=images.device)], dim=0)

        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        dt = steps[1] - steps[0]

        class func2(torch.nn.Module):
            def __init__(self, cls, h, w, max_scale):
                super().__init__()
                self.cls = cls
                self.h = h
                self.w = w
                self.max_scale = max_scale
            def __call__(self, x):
                h, w = x.shape[1:3]
                current_scale_h = (h/self.h)
                current_scale_w = (w/self.w)
                random_scale_w = max(1.0, current_scale_w)
                random_scale_h = max(1.0, current_scale_h)
                scale = torch.ones((1,1,1,1,1,2)).to(x.device, x.dtype)
                scale[..., 0] = scale[..., 0]*min(self.max_scale, random_scale_w)*6
                scale[..., 1] = scale[..., 1]*min(self.max_scale, random_scale_h)*6
                self.cls.max_scale = scale
                x = self.cls(x)
                self.cls.max_scale = 6
                return x

        for i, t_cur in enumerate(steps[:-1]):
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            t = t_cur.repeat(2)

            H, W = x.shape[2:]
            for j, block in enumerate(net.blocks):
                if j < len(net.blocks) - 4 and j > 4 and t_cur[0] < 0.6:
                    max_scale = 1 + (0.6-t_cur[0])/0.6
                else:
                    max_scale = 1
                attn = block.attn
                setattr(block, "attn", func2(attn, 16, 16, max_scale))
            out = net(cfg_x, t, labels)
            for j, block in enumerate(net.blocks):
                setattr(block, "attn", block.attn.cls)
            out = self.guidance_fn(out, self.guidance)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, self.last_step, s=s, w=w)
        return x