import lpips
from torch import nn
from torch.nn import functional as F


class LenslessLoss(nn.Module):
    def __init__(self, mse_weight=1, lpips_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.lpips = lpips.LPIPS(net="vgg")

    def forward(
        self,
        reconstruction_roi,
        target_roi,
        **batch,
    ):
        mse_loss = F.mse_loss(reconstruction_roi, target_roi)
        lpips_loss = self.lpips(
            reconstruction_roi.clamp(0, 1) * 2 - 1,
            target_roi.clamp(0, 1) * 2 - 1,
        ).mean()
        loss = self.mse_weight * mse_loss + self.lpips_weight * lpips_loss
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
        }
