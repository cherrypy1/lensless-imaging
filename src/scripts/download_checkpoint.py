import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


FILES = {
    "unrolled_admm20": "unrolled_admm20.pth",
    "leadmm5_pre": "leadmm5_pre.pth",
    "leadmm5_post": "leadmm5_post.pth",
    "leadmm5_pre_post": "leadmm5_pre_post.pth",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        choices=["all"] + list(FILES.keys()),
        default="all",
    )
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = FILES.keys() if args.name == "all" else [args.name]
    for name in names:
        path = hf_hub_download(
            repo_id="artii-ml/lensless-imaging",
            filename=FILES[name],
            local_dir=output,
        )
        print(path)


if __name__ == "__main__":
    main()
