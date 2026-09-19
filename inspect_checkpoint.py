import argparse
import torch

from src.models import GINEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/core_pipeline/graphcl_checkpoint.pt",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.path, map_location="cpu", weights_only=False)
    print(type(ckpt))
    if isinstance(ckpt, dict):
        print("Keys:", sorted(ckpt.keys()))
        if "encoder_config" in ckpt:
            print("encoder_config:", ckpt["encoder_config"])
            encoder = GINEncoder.from_checkpoint(ckpt)
            n = sum(p.numel() for p in encoder.parameters())
            print(f"Loaded GINEncoder parameters: {n:,}")
        else:
            print("No encoder_config; this is not a canonical GraphCL checkpoint.")
    else:
        print(ckpt)


if __name__ == "__main__":
    main()
