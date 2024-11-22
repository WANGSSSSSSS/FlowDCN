import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.sampling import BaseScheduler
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import List
from PIL import Image
import torch
import random
import numpy as np
import copy
import torchvision.transforms as tvtf


def center_crop_arr(pil_image, width, height):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while pil_image.size[0] >= 2 * width and pil_image.size[1] >= 2 * height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = max(width / pil_image.size[0], height / pil_image.size[1])
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = random.randint(0, (arr.shape[0] - height))
    crop_x = random.randint(0, (arr.shape[1] - width))
    return Image.fromarray(arr[crop_y: crop_y + height, crop_x: crop_x + width])

class VARCandidate:
    def __init__(self, aspect_ratio, width, height, buffer, max_buffer_size=1024):
        self.aspect_ratio = aspect_ratio
        self.width = int(width)
        self.height = int(height)
        self.buffer = buffer
        self.max_buffer_size = max_buffer_size
        self.transform = lambda x: x.float()/127.5 - 1.0
    def add_sample(self, data):
        self.buffer.append(data)
        self.buffer = self.buffer[-self.max_buffer_size:]

    def ready(self, batch_size):
        return len(self.buffer) >= batch_size

    def get_batch(self, batch_size):
        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        batch = [copy.deepcopy(b.result()) for b in batch]
        images, labels = zip(*batch)
        images = [torch.from_numpy(im).cuda() for im in images]
        images = torch.stack(images)
        labels = torch.tensor(labels).cuda().long()
        images = self.transform(images)
        return (images, labels)


def process_fn(width, height, data):
    image, label = data
    image = center_crop_arr(image, width, height)
    image = np.array(image).transpose(2, 0, 1)
    return image, label

class VARTransformEngine:
    def __init__(self,
                 base_image_size,
                 num_aspect_ratios,
                 min_aspect_ratio,
                 max_aspect_ratio,
                 num_workers = 8,
                 ):
        self.base_image_size = base_image_size
        self.num_aspect_ratios = num_aspect_ratios
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.aspect_ratios = np.linspace(self.min_aspect_ratio, self.max_aspect_ratio, self.num_aspect_ratios)
        self.aspect_ratios = self.aspect_ratios.tolist()
        self.candidates_pool = []
        for i in range(self.num_aspect_ratios):
            candidate = VARCandidate(
                aspect_ratio=self.aspect_ratios[i],
                width=int(self.base_image_size * self.aspect_ratios[i] ** 0.5 // 16 * 16),
                height=int(self.base_image_size * self.aspect_ratios[i] ** -0.5 // 16 * 16),
                buffer=[],
                max_buffer_size=1024
            )
            self.candidates_pool.append(candidate)
        self.default_candidate = VARCandidate(
            aspect_ratio=1.0,
            width=self.base_image_size,
            height=self.base_image_size,
            buffer=[],
            max_buffer_size=1024,
        )
        self.executor_pool = ProcessPoolExecutor(max_workers=num_workers)
        self.prefill = 100
    def find_candidate(self, data):
        image = data[0]
        aspect_ratio = image.size[0] / image.size[1]
        min_distance = 1000000
        min_candidate = None
        for candidate in self.candidates_pool:
            dis = abs(aspect_ratio - candidate.aspect_ratio)
            if dis < min_distance:
                min_distance = dis
                min_candidate = candidate
        return min_candidate


    def __call__(self, batch_data):
        self.prefill -= 1
        for data in batch_data:
            candidate = self.find_candidate(data)
            future = self.executor_pool.submit(process_fn, candidate.width, candidate.height, data)
            candidate.add_sample(future)
            if self.prefill >= 0:
                future = self.executor_pool.submit(process_fn,
                                                   self.default_candidate.width,
                                                   self.default_candidate.height,
                                                   data)
                self.default_candidate.add_sample(future)
        batch_size = len(batch_data)
        random.shuffle(self.candidates_pool)
        for candidate in self.candidates_pool:
            if candidate.ready(batch_size=batch_size):
                return candidate.get_batch(batch_size=batch_size)

        # fallback to default 256
        for data in batch_data:
            future = self.executor_pool.submit(process_fn,
                             self.default_candidate.width,
                             self.default_candidate.height,
                             data)
            self.default_candidate.add_sample(future)
        return self.default_candidate.get_batch(batch_size=batch_size)

class VARTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            base_image_size=256,
            num_aspect_ratios=10,
            min_aspect_ratio=0.5,
            max_aspect_ratio=3.0,
            lognorm_t=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.var_transform_engine = VARTransformEngine(
            base_image_size=base_image_size,
            num_aspect_ratios=num_aspect_ratios,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        self.lognorm_t = lognorm_t

    def preproprocess(self, x, y):
        batch = list(zip(x, y))
        batch = self.var_transform_engine(batch)
        images, labels = batch
        batch = super().preproprocess(images, labels)
        return batch

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
        loss = (out - v_t) ** 2
        out = dict(
            loss=loss.mean(),
        )
        return out
