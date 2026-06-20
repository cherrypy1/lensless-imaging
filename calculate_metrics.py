import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def load_image(path):
    image = Image.open(path).convert("RGB")
    array = np.array(image).astype(np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def resize_target(target, pred):
    if target.shape[-2:] == pred.shape[-2:]:
        return target
    image = Image.fromarray(
        (target.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    )
    image = image.resize((pred.shape[-1], pred.shape[-2]), Image.NEAREST)
    array = np.array(image).astype(np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lensed", required=True)
    parser.add_argument("--reconstructions", required=True)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lensed_dir = Path(args.lensed)
    reconstruction_dir = Path(args.reconstructions)
    metrics = [
        MSEMetric(name="mse"),
        PSNRMetric(name="psnr", device=device),
        SSIMMetric(name="ssim", device=device),
        LPIPSMetric(name="lpips", device=device),
    ]
    values = {metric.name: [] for metric in metrics}
    with torch.no_grad():
        recon_paths = sorted(reconstruction_dir.glob("*.png"))
        if args.count is not None:
            recon_paths = recon_paths[: args.count]
        for recon_path in recon_paths:
            lensed_path = lensed_dir / recon_path.name
            if not lensed_path.exists():
                continue
            reconstruction = load_image(recon_path)
            target = resize_target(load_image(lensed_path), reconstruction)
            reconstruction = reconstruction.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            for metric in metrics:
                values[metric.name].append(
                    metric(
                        reconstruction_roi=reconstruction,
                        target_roi=target,
                    )
                )

    if len(values["mse"]) == 0:
        print("нет совпавших изображений")
        return

    print(f"count: {len(values['mse'])}")
    for name, metric_values in values.items():
        print(f"{name}: {sum(metric_values) / len(metric_values):.4f}")


if __name__ == "__main__":
    main()
