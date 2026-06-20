import torch


def gradient(x):
    dh = torch.roll(x, shifts=-1, dims=-2) - x
    dw = torch.roll(x, shifts=-1, dims=-1) - x
    return torch.stack([dh, dw], dim=1)


def divergence(x):
    dh = torch.roll(x[:, 0], shifts=1, dims=-2) - x[:, 0]
    dw = torch.roll(x[:, 1], shifts=1, dims=-1) - x[:, 1]
    return dh + dw


def soft_threshold(x, value):
    return torch.sign(x) * torch.relu(torch.abs(x) - value)


def tv_denominator(shape, device, dtype):
    h, w = shape
    impulse = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    impulse[..., 0, 0] = 1
    diff_h = torch.roll(impulse, shifts=-1, dims=-2) - impulse
    diff_w = torch.roll(impulse, shifts=-1, dims=-1) - impulse
    fft_h = torch.fft.rfft2(diff_h)
    fft_w = torch.fft.rfft2(diff_w)
    return fft_h.abs() ** 2 + fft_w.abs() ** 2
