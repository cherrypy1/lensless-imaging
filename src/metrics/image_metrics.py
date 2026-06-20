import lpips
import torch
from torch.nn import functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.metrics.base_metric import BaseMetric


def prepare_images(reconstruction_roi, target_roi):
    return reconstruction_roi.clamp(0, 1), target_roi.clamp(0, 1)


def get_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


class MSEMetric(BaseMetric):
    def __call__(self, reconstruction_roi, target_roi, **batch):
        reconstruction_roi, target_roi = prepare_images(reconstruction_roi, target_roi)
        return F.mse_loss(reconstruction_roi, target_roi).item()


class PSNRMetric(BaseMetric):
    def __init__(self, device="auto", name=None):
        super().__init__(name)
        self.metric = PeakSignalNoiseRatio(data_range=1).to(get_device(device))

    def __call__(self, reconstruction_roi, target_roi, **batch):
        reconstruction_roi, target_roi = prepare_images(reconstruction_roi, target_roi)
        self.metric.reset()
        return self.metric(reconstruction_roi, target_roi).item()


class SSIMMetric(BaseMetric):
    def __init__(self, device="auto", name=None):
        super().__init__(name)
        self.metric = StructuralSimilarityIndexMeasure(data_range=1).to(
            get_device(device)
        )

    def __call__(self, reconstruction_roi, target_roi, **batch):
        reconstruction_roi, target_roi = prepare_images(reconstruction_roi, target_roi)
        self.metric.reset()
        return self.metric(reconstruction_roi, target_roi).item()


class LPIPSMetric(BaseMetric):
    def __init__(self, device="auto", name=None):
        super().__init__(name)
        self.metric = lpips.LPIPS(net="vgg").to(get_device(device))

    def __call__(self, reconstruction_roi, target_roi, **batch):
        reconstruction_roi, target_roi = prepare_images(reconstruction_roi, target_roi)
        value = self.metric(reconstruction_roi * 2 - 1, target_roi * 2 - 1)
        return value.mean().item()
