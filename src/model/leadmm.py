from torch import nn

from src.model.admm import ADMMReconstruction
from src.model.drunet import DRUNet


class LeADMM(nn.Module):
    def __init__(
        self,
        iterations=5,
        use_preprocessor=False,
        use_postprocessor=False,
        pre_channels=(32, 64, 128, 256),
        post_channels=(32, 64, 128, 256),
        mu=1e-4,
        tau=2e-4,
        pad_factor=2,
        roi_top=80,
        roi_left=100,
        roi_height=200,
        roi_width=266,
    ):
        super().__init__()
        self.preprocessor = None
        self.postprocessor = None
        if use_preprocessor:
            self.preprocessor = DRUNet(channels=pre_channels)
        if use_postprocessor:
            self.postprocessor = DRUNet(channels=post_channels)
        self.admm = ADMMReconstruction(
            iterations=iterations,
            mu=mu,
            tau=tau,
            pad_factor=pad_factor,
            roi_top=roi_top,
            roi_left=roi_left,
            roi_height=roi_height,
            roi_width=roi_width,
        )

    def forward(self, measurement, psf, **batch):
        camera_input = measurement
        if self.preprocessor is not None:
            camera_input = self.preprocessor(camera_input)
        output = self.admm(camera_input, psf)
        camera_output = output["reconstruction"]
        if self.postprocessor is not None:
            reconstruction = self.postprocessor(camera_output)
        else:
            reconstruction = camera_output
        reconstruction = reconstruction.clamp_min(0)
        output["camera_input"] = camera_input
        output["camera_output"] = camera_output
        output["reconstruction"] = reconstruction
        output["reconstruction_roi"] = self.admm.crop_roi(reconstruction)
        return output


class LeADMM5Pre(LeADMM):
    def __init__(self, **kwargs):
        super().__init__(
            use_preprocessor=True,
            **kwargs,
        )


class LeADMM5Post(LeADMM):
    def __init__(self, **kwargs):
        super().__init__(
            use_postprocessor=True,
            **kwargs,
        )


class LeADMM5PrePost(LeADMM):
    def __init__(self, **kwargs):
        super().__init__(
            use_preprocessor=True,
            use_postprocessor=True,
            pre_channels=(32, 64, 116, 128),
            post_channels=(32, 64, 116, 128),
            **kwargs,
        )
