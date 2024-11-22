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


import logging
logger = logging.getLogger(__name__)

class EulerSampler(BaseSampler):
    def __init__(
            self,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            last_step=None,
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

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

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
            out = net(cfg_x, t, labels)
            out = self.guidance_fn(out, self.guidance)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, self.last_step, s=s, w=w)
        return x


class HenuSampler(BaseSampler):
    def __init__(
            self,
            num_steps: int = 250,
            guidance=4.0,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            guidance_fn: Callable = c3_guidance_fn,
            pred_eps=False,
            exact_henu=False,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.pred_eps = pred_eps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.exact_henu = exact_henu
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        assert  self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, images, labels):
        """
        sampling process of Henu sampler
        -
        """
        batch_size = images.shape[0]
        steps = torch.linspace(0.0, 1 - self.last_step, self.num_steps, device=images.device)
        steps = torch.cat([steps, torch.tensor([1.0], device=images.device)], dim=0)
        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        v_hat, s_hat = 0.0, 0.0
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            drift_coeff = self.scheduler.drift_coefficient(t_cur)
            diffusion_coeff = self.scheduler.diffuse_coefficient(t_cur)
            alpha_over_dalpha = 1/self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)

            t_hat = t_next
            t_hat = t_hat.repeat(batch_size)
            sigma_hat = self.scheduler.sigma(t_hat)
            drift_coeff_hat = self.scheduler.drift_coefficient(t_hat)
            diffusion_coeff_hat = self.scheduler.diffuse_coefficient(t_hat)
            alpha_over_dalpha_hat = 1 / self.scheduler.dalpha_over_alpha(t_hat)
            dsigma_mul_sigma_hat = self.scheduler.dsigma_mul_sigma(t_hat)

            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0
            if i == 0 or self.exact_henu:
                cfg_x = torch.cat([x, x], dim=0)
                t_cur = t_cur.repeat(2)
                out = net(cfg_x, t_cur, labels)
                out = self.guidance_fn(out, self.guidance)

                if self.pred_eps:
                    s = out / sigma
                    v = drift_coeff * x + diffusion_coeff * s
                else:
                    v = out
                    s = ((alpha_over_dalpha)*v - x)/(sigma**2 - (alpha_over_dalpha)*dsigma_mul_sigma)
            else:
                v = v_hat
                s = s_hat
            x_hat = self.step_fn(x, v, dt, s=s, w=w)
            # henu correct
            if i < self.num_steps -1:
                cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                t_hat = t_hat.repeat(2)
                out = net(cfg_x_hat, t_hat, labels)
                out = self.guidance_fn(out, self.guidance)
                if self.pred_eps:
                    s_hat = out / sigma_hat
                    v_hat = drift_coeff_hat * x_hat + diffusion_coeff_hat * s_hat
                else:
                    v_hat = out
                    s_hat = ((alpha_over_dalpha_hat)* v_hat - x_hat) / (sigma_hat ** 2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat)
                v = (v + v_hat) / 2
                s = (s + s_hat) / 2
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, self.last_step, s=s, w=w)
        return x