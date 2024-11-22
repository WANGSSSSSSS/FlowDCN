import os.path

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import Callable, Iterable, Dict, Any

from src.utils.vae import PixelVAE
from src.utils.model_loader import ModelLoader

from src.utils.metrics import UnifiedMetric
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.saver import ImageSaver

OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]
import logging
log = logging.getLogger(__name__)

class LightningModel(pl.LightningModule):
    def __init__(self,
                 vae: PixelVAE,
                 denoiser: nn.Module,
                 metric: UnifiedMetric,
                 diffusion_trainer: BaseTrainer,
                 diffusion_sampler: BaseSampler,
                 optimizer: OptimizerCallable,
                 vae_path: str= None,
                 denoiser_path: str= None,
                 lr_scheduler:LRSchedulerCallable = None,
                 precompute_metric_data: Dict[str, str] = None,
                 save_val_image:bool = True,
                 save_dir: str = "val",
                 ):
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.metric = metric

        self.vae_path = vae_path
        self.denoiser_path = denoiser_path

        self.model_loader = ModelLoader()
        self.diffusion_trainer = diffusion_trainer
        self.diffusion_sampler = diffusion_sampler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.precompute_metric_data = precompute_metric_data


        self._strict_loading = False
        self.save_val_image = save_val_image
        self.save_dir = save_dir

    def configure_model(self) -> None:
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.model_loader.prepare(rank, world_size, self.device, self.vae_path, self.denoiser_path, self.precompute_metric_data, self.dtype)
        self.vae, self.denoiser, self.metric = self.model_loader.load(self.vae, self.denoiser, self.metric)
        self.diffusion_trainer.setup(self.vae)
        # disable grad for metric and vae
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.metric.parameters():
            p.requires_grad = False

        # recheck metric fid
        reset = self.precompute_metric_data is None
        assert reset == self.metric.reset_real_features

        if not os.path.isabs(self.save_dir):
             # not a relative path
             self.save_dir = os.path.join(self.trainer.default_root_dir, self.save_dir)

    def on_fit_start(self) -> None:
        if self.trainer.strategy.is_global_zero :
            os.makedirs(self.trainer.default_root_dir, exist_ok=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(self.denoiser.parameters())
        if self.lr_scheduler is None:
            return dict(
                optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.diffusion_trainer(self.denoiser, x, y)
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss["loss"]

    def on_validation_epoch_start(self) -> None:
        if self.precompute_metric_data is not None:
            self.metric.reset()
        self.max_save_images_num = 1000
        # setup image saver
        save_dir = os.path.join(self.save_dir, f"epoch_{self.current_epoch}")
        self.image_saver = ImageSaver(save_dir,
                rank=self.trainer.global_rank,
                max_save_num=self.max_save_images_num
        )

    def validation_step(self, batch, batch_idx):
        # insert epoch info to metadata path
        x, y, meta_data = batch
        new_meta_data = meta_data
        batch = (x, y, new_meta_data)
        samples = self.predict_step(batch, batch_idx, save=self.save_val_image)
        return samples

    def on_validation_epoch_end(self) -> None:
        exit()
        self.trainer.strategy.barrier()
        if self.precompute_metric_data is not None:
            metric_dict = self.metric.compute()
            self.log_dict(metric_dict, prog_bar=True, sync_dist=True, on_epoch=True)
        self.image_saver.upload_all()
        del self.image_saver


    def on_predict_epoch_start(self) -> None:
        self.max_save_images_num = 500000
        self.image_saver = ImageSaver(self.save_dir,
            rank=self.trainer.global_rank,
            max_save_num=self.max_save_images_num
        )

    def predict_step(self, batch, batch_idx, save=True):
        zT, y, metadata = batch

        # Sample images:
        samples = self.diffusion_sampler(self.denoiser, zT, y)
        # samples = samples.to(self.vae.dtype)
        samples = self.vae.decode(samples)

        if save:
            if self.trainer.is_global_zero:
                self.image_saver.upload_image(samples, metadata)
            self.image_saver.save_image(samples, metadata)

        # predict and eval need metric when precompute data enabled
        if self.precompute_metric_data is not None:
            # fp32 -1,1 -> uint8 0,255
            samples = torch.clip_((samples+1)*127.5 + 0.5, 0, 255).to(torch.uint8)
            self.metric.update(samples, False)
        return samples

    def on_predict_epoch_end(self) -> None:
        self.trainer.strategy.barrier()
        if self.precompute_metric_data is not None:
            metric_dict = self.metric.compute()
            for k, v in metric_dict.items():
                print(f"{k}:{v.item()}")
        self.image_saver.upload_all()
        del self.image_saver

    def on_test_start(self) -> None:
        self.metric.reset()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch
        if self.precompute_metric_data is not None:
            self.metric.update(images, False)
        else:
            self.metric.update(images, dataloader_idx)
    def on_test_epoch_end(self) -> None:
        metric = self.metric.compute()
        self.log_dict(metric, on_epoch=True, prog_bar=True, on_step=False)

