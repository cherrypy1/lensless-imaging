import argparse
from pathlib import Path
from urllib.request import urlretrieve


def get_files(split):
    count = 9 if split == "train" else 2
    return [f"{split}-{i:05d}-of-{count:05d}.parquet" for i in range(count)]


def download_split(output, split):
    base_url = (
        "https://huggingface.co/datasets/"
        "bezzam/DigiCam-Mirflickr-MultiMask-10K/resolve/main/data"
    )
    for name in get_files(split):
        path = output / name
        if path.exists():
            print(path)
            continue
        urlretrieve(f"{base_url}/{name}", path)
        print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/digicam")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        download_split(output, split)


if __name__ == "__main__":
    main()
