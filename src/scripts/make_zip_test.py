import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from datasets import load_dataset

from src.datasets.digicam import get_data_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/digicam")
    parser.add_argument("--mask-dir", default="data/masks")
    parser.add_argument("--output", default="custom_test.zip")
    args = parser.parse_args()

    output = Path(args.output)
    files = get_data_files("test", args.data_dir)
    dataset = load_dataset("parquet", data_files={"test": files})["test"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ["lensless", "masks", "lensed"]:
            (root / name).mkdir()

        for i, item in enumerate(dataset):
            image_id = f"ImageID{i + 1}"
            item["lensless"].save(root / "lensless" / f"{image_id}.png")
            item["lensed"].save(root / "lensed" / f"{image_id}.png")
            mask = Path(args.mask_dir) / f"mask_{item['mask_label']}.npy"
            shutil.copyfile(mask, root / "masks" / f"{image_id}.npy")

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                archive.write(path, path.relative_to(root))

    print(output)


if __name__ == "__main__":
    main()
