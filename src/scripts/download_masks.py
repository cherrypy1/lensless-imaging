import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/masks")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=99)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_url = (
        "https://huggingface.co/datasets/"
        "bezzam/DigiCam-Mirflickr-MultiMask-10K/resolve/main/masks"
    )
    for i in range(args.first, args.last + 1):
        url = f"{base_url}/mask_{i}.npy"
        path = output / f"mask_{i}.npy"
        urlretrieve(url, path)
        print(path)


if __name__ == "__main__":
    main()
