import torch
from torch import nn

from src.model.fft import (
    center_crop,
    center_pad,
    convolve_fft,
    crop_mask,
    padded_shape,
    psf_fft,
    transpose_convolve_fft,
)
from src.model.tv import divergence, gradient, soft_threshold, tv_denominator


class ADMMParameters(nn.Module):
    def __init__(self, iterations, mu=1e-4, tau=2e-4, learnable=False):
        super().__init__()
        self.iterations = iterations
        self.learnable = learnable
        values = torch.stack(
            [
                torch.log(torch.tensor(mu).repeat(iterations)),
                torch.log(torch.tensor(mu).repeat(iterations)),
                torch.log(torch.tensor(mu).repeat(iterations)),
                torch.log(torch.tensor(tau).repeat(iterations)),
            ],
            dim=1,
        )
        if learnable:
            self.values = nn.Parameter(values)
        else:
            self.register_buffer("values", values)

    def forward(self, index, dtype, device):
        values = torch.exp(self.values[index]).to(device=device, dtype=dtype)
        return values[0], values[1], values[2], values[3]


class ADMMReconstruction(nn.Module):
    def __init__(
        self,
        iterations=20,
        mu=1e-4,
        tau=2e-4,
        learnable=True,
        pad_factor=2,
        roi_top=80,
        roi_left=100,
        roi_height=200,
        roi_width=266,
        psf_scale=5,
    ):
        super().__init__()
        self.iterations = iterations
        self.pad_factor = pad_factor
        self.roi_top = roi_top
        self.roi_left = roi_left
        self.roi_height = roi_height
        self.roi_width = roi_width
        self.psf_scale = psf_scale
        self.params = ADMMParameters(iterations, mu=mu, tau=tau, learnable=learnable)

    def forward(self, measurement, psf, **batch):
        image_shape = measurement.shape[-2:]
        shape = padded_shape(image_shape, self.pad_factor)
        data = self.prepare_data(measurement, psf, shape)
        x = self.run_admm(data)
        reconstruction = center_crop(x, image_shape)
        return {
            "reconstruction": reconstruction,
            "reconstruction_roi": self.crop_roi(reconstruction),
        }

    def prepare_data(self, measurement, psf, shape):
        b = center_pad(measurement, shape)
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-13)
        psf = psf * self.psf_scale
        h_fft = psf_fft(psf, shape)
        mask = crop_mask(
            measurement.shape[-2:],
            shape,
            measurement.device,
            measurement.dtype,
        )
        ctb = b
        denom_tv = tv_denominator(shape, measurement.device, measurement.dtype)
        return b, h_fft, mask, ctb, denom_tv

    def run_admm(self, data):
        b, h_fft, mask, ctb, denom_tv = data
        x = torch.zeros_like(b)
        v = torch.zeros_like(b)
        w = torch.zeros_like(b)
        u = torch.zeros(b.shape[0], 2, *b.shape[1:], device=b.device, dtype=b.dtype)
        alpha1 = torch.zeros_like(b)
        alpha2 = torch.zeros_like(u)
        alpha3 = torch.zeros_like(b)
        for i in range(self.iterations):
            mu1, mu2, mu3, tau = self.params(i, b.dtype, b.device)
            h_x = convolve_fft(x, h_fft)
            u = soft_threshold(gradient(x) + alpha2 / mu2, tau)
            v = (alpha1 + mu1 * h_x + ctb) / (mask + mu1)
            w = torch.relu(alpha3 / mu3 + x)

            part_w = mu3 * w - alpha3
            part_u = divergence(mu2 * u - alpha2)
            part_v = transpose_convolve_fft(mu1 * v - alpha1, h_fft)
            rhs = part_w + part_u + part_v

            denom = mu1 * h_fft.abs() ** 2 + mu2 * denom_tv + mu3
            x_fft = torch.fft.rfft2(rhs) / denom
            x = torch.fft.irfft2(x_fft, s=b.shape[-2:])
            h_x = convolve_fft(x, h_fft)
            alpha1 = alpha1 + mu1 * (h_x - v)
            alpha2 = alpha2 + mu2 * (gradient(x) - u)
            alpha3 = alpha3 + mu3 * (x - w)
        return x

    def crop_roi(self, reconstruction):
        return reconstruction[
            ...,
            self.roi_top : self.roi_top + self.roi_height,
            self.roi_left : self.roi_left + self.roi_width,
        ]


class ADMM100(ADMMReconstruction):
    def __init__(self, **kwargs):
        super().__init__(iterations=100, learnable=False, **kwargs)


class UnrolledADMM20(ADMMReconstruction):
    def __init__(self, **kwargs):
        super().__init__(iterations=20, **kwargs)
