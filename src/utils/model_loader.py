from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from lightning.fabric.utilities.types import _PATH


import logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self,):
        super().__init__()
    def prepare(self, rank, world_size, device, vae_path, denoiser_path, precompute_data_path, dtype):
        self._device = device
        self._vae_path = vae_path
        self._denoiser_path = denoiser_path
        self._precompute_data_path = precompute_data_path
        self._dtype = dtype  # not used

        self.rank = rank
        self.world_size = world_size
        if self._precompute_data_path:
            _precompute_data_path = dict()
            for k, v in self._precompute_data_path.items():
                    _precompute_data_path[k] = v
            self._precompute_data_path = _precompute_data_path

    def load(self, vae, denoiser, metric):
        if self._vae_path:
           vae = vae.from_pretrained(self._vae_path).to(self._device)
        if self._denoiser_path:
            weight = torch.load(self._denoiser_path, map_location=torch.device('cpu'))
            # import pdb; pdb.set_trace()
            if "optimizer_states" in weight.keys():
                weight = weight["optimizer_states"][0]["ema"]
                params = list(denoiser.parameters())
                for w,p in zip(weight, params):
                    p.data.copy_(w)
            else:
                denoiser.load_state_dict(weight)
            denoiser.to(self._device)
        if self._precompute_data_path:
            metric.load_precompute_data(self._precompute_data_path, self.rank, self.world_size)
        return vae, denoiser, metric