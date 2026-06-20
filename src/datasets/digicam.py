from pathlib import Path

from datasets import load_dataset

from lensless_helpers.preprocessor import get_dataset_object, get_roi
from src.datasets.image_utils import chw, load_mask, psf_chw


def get_split_count(split):
    if split == "train":
        return 9
    return 2


def get_data_files(split, data_dir):
    data_dir = Path(data_dir)
    count = get_split_count(split)
    paths = [
        data_dir / f"{split}-{i:05d}-of-{count:05d}.parquet"
        for i in range(count)
    ]
    return [str(path) for path in paths]


class DigiCamDataset:
    def __init__(
        self,
        split,
        mask_dir,
        data_dir="data/digicam",
        data_files=None,
        limit=None,
        instance_transforms=None,
    ):
        self.split = split
        self.mask_dir = Path(mask_dir)
        self.limit = limit
        self.instance_transforms = instance_transforms
        if data_files is None:
            data_files = get_data_files(split, data_dir)
        self.dataset = load_dataset("parquet", data_files={split: data_files})[split]

    def __len__(self):
        if self.limit is None:
            return len(self.dataset)
        return min(self.limit, len(self.dataset))

    def __getitem__(self, index):
        item = self.dataset[index]
        mask = load_mask(self.mask_dir / f"mask_{item['mask_label']}.npy")
        target, measurement, psf = get_dataset_object(
            item["lensed"],
            item["lensless"],
            mask,
        )
        target_roi = get_roi(target)
        data = {
            "measurement": chw(measurement),
            "psf": psf_chw(psf),
            "target": chw(target),
            "target_roi": chw(target_roi),
            "id": str(item.get("image_id", index)),
        }
        return self.preprocess(data)

    def preprocess(self, data):
        if self.instance_transforms is not None:
            for name in self.instance_transforms.keys():
                data[name] = self.instance_transforms[name](data[name])
        return data
