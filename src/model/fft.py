import math
import torch


def padded_shape(shape, factor):
    h, w = shape
    return math.ceil(h * factor), math.ceil(w * factor)


def center_slices(small_shape, big_shape):
    h, w = small_shape
    big_h, big_w = big_shape
    top = (big_h - h) // 2
    left = (big_w - w) // 2
    return slice(top, top + h), slice(left, left + w)


def center_pad(x, shape):
    result = x.new_zeros(x.shape[:-2] + shape)
    rows, cols = center_slices(x.shape[-2:], shape)
    result[..., rows, cols] = x
    return result


def center_crop(x, shape):
    rows, cols = center_slices(shape, x.shape[-2:])
    return x[..., rows, cols]


def crop_mask(image_shape, padded_shape, device, dtype):
    mask = torch.zeros(1, 1, *padded_shape, device=device, dtype=dtype)
    rows, cols = center_slices(image_shape, padded_shape)
    mask[..., rows, cols] = 1
    return mask


def psf_fft(psf, shape):
    result = psf.new_zeros(psf.shape[:-2] + shape)
    rows, cols = center_slices(psf.shape[-2:], shape)
    result[..., rows, cols] = psf
    psf = result
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    return torch.fft.rfft2(psf)


def convolve_fft(x, kernel_fft):
    x_fft = torch.fft.rfft2(x)
    y_fft = x_fft * kernel_fft
    return torch.fft.irfft2(y_fft, s=x.shape[-2:])


def transpose_convolve_fft(x, kernel_fft):
    x_fft = torch.fft.rfft2(x)
    y_fft = x_fft * kernel_fft.conj()
    return torch.fft.irfft2(y_fft, s=x.shape[-2:])
