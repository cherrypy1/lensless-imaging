from pathlib import Path

from lensless_helpers.preprocessor import get_dataset_object, get_roi
from src.datasets.image_utils import (
    chw,
    find_image,
    load_mask,
    load_rgb,
    psf_chw,
)


class CustomDirDataset:
    def __init__(self, root, instance_transforms=None):
        self.root = Path(root)
        self.lensless_dir = self.root / "lensless"
        self.mask_dir = self.root / "masks"
        self.lensed_dir = self.root / "lensed"
        self.instance_transforms = instance_transforms
        self.files = sorted(self.lensless_dir.glob("*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        lensless_path = self.files[index]
        image_id = lensless_path.stem
        lensless = load_rgb(lensless_path)
        mask = load_mask(self.mask_dir / f"{image_id}.npy")
        lensed_path = find_image(self.lensed_dir / image_id)
        if lensed_path is None:
            target, measurement, psf = get_dataset_object(lensless, lensless, mask)
            data = {
                "measurement": chw(measurement),
                "psf": psf_chw(psf),
                "id": image_id,
            }
            return self.preprocess(data)
        lensed = load_rgb(lensed_path)
        target, measurement, psf = get_dataset_object(lensed, lensless, mask)
        target_roi = get_roi(target)
        data = {
            "measurement": chw(measurement),
            "psf": psf_chw(psf),
            "target": chw(target),
            "target_roi": chw(target_roi),
            "id": image_id,
        }
        return self.preprocess(data)

    def preprocess(self, data):
        if self.instance_transforms is not None:
            for name in self.instance_transforms.keys():
                data[name] = self.instance_transforms[name](data[name])
        return data
