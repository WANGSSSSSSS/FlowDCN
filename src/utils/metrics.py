from torchmetrics import Metric
from torchmetrics import MetricCollection
from typing import Union, Any, List, Optional, Tuple, Dict
import torch
import numpy
from torch import Tensor
from torch.nn import Module
from copy import deepcopy
from torchmetrics.image import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3, interpolate_bilinear_2d_like_tensorflow1x
from torch.nn.functional import adaptive_avg_pool2d

class InceptionV3(FeatureExtractorInceptionV3):
    """Module that never leaves evaluation mode."""

    def __init__(
        self,
        name: str,
        features_list: Tuple[str],
    ) -> None:

        super().__init__(name, features_list)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> "InceptionV3":
        """Force network to always be in evaluation mode."""
        return super().train(False)
    @staticmethod
    def get_provided_features_list():
        return "logits","unbiased_logits", "pool", "spatial"

    @torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True)
    def forward(self, x: Tensor) -> Dict:
        """Forward method of inception net.

        Copy of the forward method from this file:
        https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/feature_extractor_inceptionv3.py
        with a single line change regarding the casting of `x` in the beginning.

        Corresponding license file (Apache License, Version 2.0):
        https://github.com/toshas/torch-fidelity/blob/master/LICENSE.md

        """
        assert(torch.is_tensor(x) and x.dtype == torch.uint8)
        features = {}
        remaining_features = self.features_list.copy()

        x = x.float()
        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        x = (x - 128) / 128

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        if "spatial" in remaining_features:
            features["spatial"] = x[:, :7, :, :].flatten(1).contiguous()
            remaining_features.remove("spatial")

        x = self.Mixed_6e(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)

        if "pool" in remaining_features:
            features["pool"] = x
            remaining_features.remove("pool")


        if "unbiased_logits" in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features["unbiased_logits"] = x
            remaining_features.remove("unbiased_logits")
            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        if "logits" in remaining_features:
            features["logits"] = x
        return features



class UnifiedMetric(Metric):
    higher_is_better: bool = None
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = None

    # fid 2048
    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    # sfid
    spatial_real_features_sum: Tensor
    spatial_real_features_cov_sum: Tensor
    spatial_real_features_num_samples: Tensor

    spatial_fake_features_sum: Tensor
    spatial_fake_features_cov_sum: Tensor
    spatial_fake_features_num_samples: Tensor

    # is
    is_features: List

    # pr & recall & kid
    real_features: List
    fake_features: List

    inception: Module
    feature_network: str = "inception"

    def __init__(
            self,
            enabled_metrics: List[str],
            reset_real_features: bool = True,
            normalize: bool = False,
    ) -> None:
        super().__init__()
        self.enabled_metrics = enabled_metrics
        features_list = list(InceptionV3.get_provided_features_list())
        self.inception = InceptionV3(name="inception-v3-compat", features_list=features_list)
        self.reset_real_features = reset_real_features
        self.normalize = normalize

        # fid
        self.add_state("real_features_sum", torch.zeros(2048).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros((2048, 2048)).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(2048).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros((2048, 2048)).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        # sfid
        self.add_state("spatial_real_features_sum", torch.zeros(2023).double(), dist_reduce_fx="sum")
        self.add_state("spatial_real_features_cov_sum", torch.zeros((2023, 2023)).double(), dist_reduce_fx="sum")
        self.add_state("spatial_real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("spatial_fake_features_sum", torch.zeros(2023).double(), dist_reduce_fx="sum")
        self.add_state("spatial_fake_features_cov_sum", torch.zeros((2023, 2023)).double(), dist_reduce_fx="sum")
        self.add_state("spatial_fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        # is
        self.add_state("is_features", [], dist_reduce_fx=None)

        # pr & recall & kid
        # self.add_state("real_features", [], dist_reduce_fx=None)
        # self.add_state("fake_features", [], dist_reduce_fx=None)
        # keeps these features only on cpu
        self.real_features = []
        self.fake_features = []

    def update_fid(self, features: Tensor, real: bool):
        features = features.double()
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]
    def compute_fid(self):
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
            a = (mu1 - mu2).square().sum(dim=-1)
            b = sigma1.trace() + sigma2.trace()
            # c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
            c = torch.from_numpy(numpy.linalg.eigvals(sigma1.cpu().numpy() @ sigma2.cpu().numpy())).sqrt().real.sum(dim=-1)
            return a + b - 2 * c

        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)
    def load_precompute_fid(self, precompute_data_path, rank, world_size):
        precompute_data = torch.load(precompute_data_path)
        self.real_features_sum = precompute_data["real_features_sum"] / world_size
        self.real_features_cov_sum = precompute_data["real_features_cov_sum"] / world_size
        self.real_features_num_samples = precompute_data["real_features_num_samples"] / world_size

    def update_spatial_feature(self, features: Tensor, real: bool):
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.spatial_real_features_sum += features.sum(dim=0)
            self.spatial_real_features_cov_sum += features.t().mm(features)
            self.spatial_real_features_num_samples += features.shape[0]
        else:
            self.spatial_fake_features_sum += features.sum(dim=0)
            self.spatial_fake_features_cov_sum += features.t().mm(features)
            self.spatial_fake_features_num_samples += features.shape[0]
    def compute_sfid(self):
        def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
            a = (mu1 - mu2).square().sum(dim=-1)
            b = sigma1.trace() + sigma2.trace()
            # c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
            c = torch.from_numpy(numpy.linalg.eigvals(sigma1.cpu().numpy() @ sigma2.cpu().numpy())).sqrt().real.sum(dim=-1)
            return a + b - 2 * c

        if self.spatial_real_features_num_samples < 2 or self.spatial_fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.spatial_real_features_sum / self.spatial_real_features_num_samples).unsqueeze(0)
        mean_fake = (self.spatial_fake_features_sum / self.spatial_fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.spatial_real_features_cov_sum - self.spatial_real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.spatial_real_features_num_samples - 1)
        cov_fake_num = self.spatial_fake_features_cov_sum - self.spatial_fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.spatial_fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)
    def load_precompute_sfid(self, precompute_data_path, rank, world_size):
        precompute_data = torch.load(precompute_data_path)
        self.spatial_real_features_sum = precompute_data["spatial_real_features_sum"] / world_size
        self.spatial_real_features_cov_sum = precompute_data["spatial_real_features_cov_sum"] / world_size
        self.spatial_real_features_num_samples = precompute_data["spatial_real_features_num_samples"] / world_size

    def update_is(self, features: Tensor, real: bool):
        if not real:
            self.is_features.append(features)
    def compute_is(self):
        """Compute metric."""
        features = dim_zero_cat(self.is_features)

        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(10, dim=0)
        log_prob = log_prob.chunk(10, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()

    def update_pool_features(self, features: Tensor, real: bool):
        features = features.float()
        if real:
            self.real_features.append(features.cpu())
        else:
            self.fake_features.append(features.cpu())
    def compute_prc(self):
        def calc_cdist_part(features_1, features_2, batch_size=5000):
            dists = []
            for feat2_batch in features_2.split(batch_size):
                dists.append(torch.cdist(features_1, feat2_batch).cpu())
            return torch.cat(dists, dim=1)

        def calculate_precision_recall_part(features_1, features_2, neighborhood=3, batch_size=5000):
            # Precision
            dist_nn_1 = []
            for feat_1_batch in features_1.split(batch_size):
                dist_nn_1.append(
                    calc_cdist_part(feat_1_batch, features_1, batch_size).kthvalue(neighborhood + 1).values)
            dist_nn_1 = torch.cat(dist_nn_1)
            precision = []
            for feat_2_batch in features_2.split(batch_size):
                dist_2_1_batch = calc_cdist_part(feat_2_batch, features_1, batch_size)
                precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
            precision = torch.cat(precision).mean().item()
            # Recall
            dist_nn_2 = []
            for feat_2_batch in features_2.split(batch_size):
                dist_nn_2.append(
                    calc_cdist_part(feat_2_batch, features_2, batch_size).kthvalue(neighborhood + 1).values)
            dist_nn_2 = torch.cat(dist_nn_2)
            recall = []
            for feat_1_batch in features_1.split(batch_size):
                dist_1_2_batch = calc_cdist_part(feat_1_batch, features_2, batch_size)
                recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
            recall = torch.cat(recall).mean().item()
            return precision, recall
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)
        # not enough memory
        return calculate_precision_recall_part(real_features, fake_features)
    def load_precompute_prc(self, precompute_data_path, rank, world_size):
        precompute_data = torch.load(precompute_data_path)
        self.real_features = precompute_data["real_features"]

    def compute_kid(self):
        def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
            """Adapted from `KID Score`_."""
            m = k_xx.shape[0]

            diag_x = torch.diag(k_xx)
            diag_y = torch.diag(k_yy)

            kt_xx_sums = k_xx.sum(dim=-1) - diag_x
            kt_yy_sums = k_yy.sum(dim=-1) - diag_y
            k_xy_sums = k_xy.sum(dim=0)

            kt_xx_sum = kt_xx_sums.sum()
            kt_yy_sum = kt_yy_sums.sum()
            k_xy_sum = k_xy_sums.sum()

            value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
            value -= 2 * k_xy_sum / (m ** 2)
            return value

        def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None,
                        coef: float = 1.0) -> Tensor:
            """Adapted from `KID Score`_."""
            if gamma is None:
                gamma = 1.0 / f1.shape[1]
            return (f1 @ f2.T * gamma + coef) ** degree

        def poly_mmd(
                f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
        ) -> Tensor:
            """Adapted from `KID Score`_."""
            k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
            k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
            k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
            return maximum_mean_discrepancy(k_11, k_12, k_22)

        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        n_samples_real = real_features.shape[0]
        n_samples_fake = fake_features.shape[0]
        kid_scores_ = []
        for _ in range(100):
            perm = torch.randperm(n_samples_real)
            f_real = real_features[perm[: 1000]]
            perm = torch.randperm(n_samples_fake)
            f_fake = fake_features[perm[: 1000]]
            o = poly_mmd(f_real, f_fake, 3, None, 1.0)
            kid_scores_.append(o)
        kid_scores = torch.stack(kid_scores_)
        return kid_scores.mean(), kid_scores.std(unbiased=False)

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        if "fid" in self.enabled_metrics:
            self.update_fid(features["pool"], real)
        if "sfid" in self.enabled_metrics:
            self.update_spatial_feature(features["spatial"], real)
        if "is" in self.enabled_metrics:
            self.update_is(features["unbiased_logits"], real)
        if "prc" in self.enabled_metrics or "kid" in self.enabled_metrics:
            self.update_pool_features(features["pool"], real)


    def compute(self) -> Dict:
        out = dict()
        # # sfid
        # _a = dict(
        #    spatial_real_features_sum = self.spatial_real_features_sum.cpu(),
        #    spatial_real_features_cov_sum = self.spatial_real_features_cov_sum.cpu(),
        #    spatial_real_features_num_samples = self.spatial_real_features_num_samples.cpu(),
        # )
        # torch.save(_a, "aa_imagenet_sfid_train512.pt")
        # # fid
        # _a = dict(
        #     real_features_sum = self.real_features_sum.cpu(),
        #     real_features_cov_sum = self.real_features_cov_sum.cpu(),
        #     real_features_num_samples = self.real_features_num_samples.cpu(),
        # )
        # torch.save(_a, "aa_imagenet_fid_train512.pt")
        if "fid" in self.enabled_metrics:
            out["fid"] = self.compute_fid()
        if "sfid" in self.enabled_metrics:
            out["sfid"] = self.compute_sfid()
        if "is" in self.enabled_metrics:
            is_mean, is_std = self.compute_is()
            out["is_mean"] = is_mean
            out["is_std"] = is_std
        if "prc" in self.enabled_metrics:
            pr, recall = self.compute_prc()
            out["precision"] = pr
            out["recall"] = recall
        if "kid" in self.enabled_metrics:
            kid_mean, kid_std = self.compute_kid()
            out["kid_mean"] = kid_mean
            out["kid_std"] = kid_std
        return out


    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            # fid
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            # sfid
            spatial_real_features_sum = deepcopy(self.spatial_real_features_sum)
            spatial_real_features_cov_sum = deepcopy(self.spatial_real_features_cov_sum)
            spatial_real_features_num_samples = deepcopy(self.spatial_real_features_num_samples)
            # pr recall
            real_features = deepcopy(self.real_features)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
            self.spatial_real_features_sum = spatial_real_features_sum
            self.spatial_real_features_cov_sum = spatial_real_features_cov_sum
            self.spatial_real_features_num_samples = spatial_real_features_num_samples
            self.real_features = real_features
        else:
            super().reset()

    def load_precompute_data(self, precompute_data_path: Dict, rank, world_size):
        if self.reset_real_features is False:
            _enabled_metrics = []
            if "fid" in precompute_data_path.keys() and "fid" in self.enabled_metrics:
                self.load_precompute_fid(precompute_data_path["fid"], rank, world_size)
                _enabled_metrics.append("fid")
            if "sfid" in precompute_data_path.keys() and "sfid" in self.enabled_metrics:
                self.load_precompute_sfid(precompute_data_path["sfid"], rank, world_size)
                _enabled_metrics.append("sfid")
            if "prc" in precompute_data_path.keys() and "prc" in self.enabled_metrics:
                self.load_precompute_prc(precompute_data_path["prc"], rank, world_size)
                _enabled_metrics.append("prc")
            if "kid" in precompute_data_path.keys() and "kid" in self.enabled_metrics:
                self.load_precompute_prc(precompute_data_path["kid"], rank, world_size)
                _enabled_metrics.append("prc")
            if "is" in self.enabled_metrics:
                _enabled_metrics.append("is")
            self.enabled_metrics = _enabled_metrics

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        out = super().set_dtype(dst_type)
        if isinstance(out.inception, InceptionV3):
            out.inception._dtype = dst_type
        return out