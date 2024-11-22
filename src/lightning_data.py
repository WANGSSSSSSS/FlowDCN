from typing import Callable, Any
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from src.data.transforms import UnifiedTransforms, xymetacollate
from src.data.randn import RandomNDataset
from src.data.metric_dataset import ImageDataset

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root,
                 test_nature_root,
                 test_gen_root,
                train_image_size=64,
                train_batch_size=64,
                train_num_workers=4,
                train_prefetch_factor=16,
                train_dataset: str = None,
                eval_batch_size=16,
                eval_num_workers=4,
                eval_max_num_instances=32,
                eval_seeds="0,1,2,3,4",
                eval_selected_classes=(207, 360, 387, 974, 88, 979, 417, 279),
                pred_batch_size=16,
                pred_num_workers=4,
                pred_seeds:str=None,
                pred_selected_classes=None,
                test_only_gen_data:Any=None,
                test_batch_size=64,
                test_num_workers=4,
                test_image_size=(224, 224),
                num_classes=1000,
                latent_shape=(4,64,64),
    ):
        super().__init__()
        eval_seeds = list(map(lambda x: int(x), eval_seeds.strip().split(","))) if eval_seeds is not None else None
        pred_seeds = list(map(lambda x: int(x), pred_seeds.strip().split(","))) if pred_seeds is not None else None

        self.train_root = train_root
        self.train_image_size = train_image_size
        self.train_dataset = train_dataset
        # stupid data_convert override, just to make nebular happy
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor

        self.test_nature_root = test_nature_root
        self.test_gen_root = test_gen_root
        self.eval_max_num_instances = eval_max_num_instances
        self.eval_seeds = eval_seeds
        self.pred_seeds = pred_seeds
        self.num_classes = num_classes
        self.latent_shape = latent_shape
        self.test_image_size = test_image_size

        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.pred_batch_size = pred_batch_size

        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers
        self.test_num_workers = test_num_workers

        self.eval_selected_classes = eval_selected_classes
        self.pred_selected_classes = pred_selected_classes

        self.test_only_gen_data = test_only_gen_data

        self._train_dataloader = None

    def prepare_data(self) -> None:
        ...

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch


    def setup(self, stage: str) -> None:
        if stage == "fit":
           assert self.train_dataset is not None
           if self.train_dataset == "imagenet":
               from src.data.imagenet import LocalDataset
               self.train_dataset = LocalDataset(
                   root=self.train_root,
                   data_convert=UnifiedTransforms(
                       size=self.train_image_size,
                   ),
               )
           elif self.train_dataset == "imagenet_cache":
               import os
               from src.data.imagenet import LocalCachedDataset
               self.train_dataset = LocalCachedDataset(
                   root=self.train_root,
               )
           elif self.train_dataset == "cifar10":
                from src.data.cifar10 import LocalDataset
                self.train_dataset = LocalDataset(
                    root=self.train_root,
                    data_convert=UnifiedTransforms(
                        size=self.train_image_size,
                    ),
                )
           elif self.train_dataset == "cifar100":
               from src.data.cifar100 import LocalDataset
               self.train_dataset = LocalDataset(
                   root=self.train_root,
                   data_convert=UnifiedTransforms(
                       size=self.train_image_size,
                   ),
               )
           elif self.train_dataset == "celeba":
               from src.data.celeba import LocalDataset
               self.train_dataset = LocalDataset(
                    root=self.train_root,
                    data_convert=UnifiedTransforms(
                        size=self.train_image_size,
                    ),
                )
           else:
             raise NotImplementedError("no such dataset")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        def collate_fn(batch):
            x, y = zip(*batch)
            if isinstance(x[0], torch.Tensor):
                x = torch.stack(x)
            if isinstance(y[0], torch.Tensor):
                y = torch.stack(y)
            else:
                y = torch.tensor(y)
            return x, y

        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        self._train_dataloader = DataLoader(
            self.train_dataset,
            self.train_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        return self._train_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.test_nature_dataset = ImageDataset(
            root=self.test_nature_root,
            image_size=self.test_image_size
        )
        self.test_gen_dataset = ImageDataset(
            root=self.test_gen_root,
            image_size=self.test_image_size
        )
        from torch.utils.data import DistributedSampler
        test_nature_sampler = DistributedSampler(self.test_nature_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        test_gen_sampler = DistributedSampler(self.test_gen_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        from src.data.metric_dataset import test_collate
        if self.test_only_gen_data:
            return [
                DataLoader(self.test_gen_dataset, self.test_batch_size, num_workers=self.test_num_workers, prefetch_factor=2, collate_fn=test_collate, sampler=test_gen_sampler),
            ]
        return [
            DataLoader(self.test_gen_dataset, self.test_batch_size, num_workers=self.test_num_workers, prefetch_factor=2, collate_fn=test_collate, sampler=test_gen_sampler),
            DataLoader(self.test_nature_dataset, self.test_batch_size, num_workers=self.test_num_workers, prefetch_factor=2, collate_fn=test_collate, sampler=test_nature_sampler),
        ]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.eval_dataset = RandomNDataset(
            seeds=self.eval_seeds,
            latent_shape=self.latent_shape,
            max_num_instances=self.eval_max_num_instances,
            num_classes=self.num_classes,
            selected_classes=self.eval_selected_classes,
        )
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.eval_dataset, self.eval_batch_size,
                          num_workers=self.eval_num_workers,
                          prefetch_factor=4,
                          collate_fn=xymetacollate,
                          sampler=sampler
                )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.pred_dataset = RandomNDataset(
            seeds= self.pred_seeds,
            max_num_instances=50000,
            num_classes=self.num_classes,
            selected_classes=self.pred_selected_classes,
            latent_shape=self.latent_shape,
        )
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
                          num_workers=self.pred_num_workers,
                          prefetch_factor=4,
                          collate_fn=xymetacollate,
                          sampler=sampler
               )
