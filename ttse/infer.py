"""Extract a target speaker from mono 16 kHz audio."""
import argparse
import soundfile as sf
import torch
from .checkpoints import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("ckpt", "mix", "enroll", "out"):
        ap.add_argument(f"--{key}", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    def read(path):
        x, sr = sf.read(path, dtype="float32")
        if sr != 16000 or x.ndim != 1:
            raise ValueError("Input must be mono 16 kHz audio")
        return torch.from_numpy(x).unsqueeze(0).to(args.device)
    model = load_model(args.ckpt, device=args.device)
    with torch.inference_mode():
        y = model.forward_streaming(read(args.mix), read(args.enroll))
    sf.write(args.out, y[0].cpu().numpy(), 16000, subtype="FLOAT")


if __name__ == "__main__":
    main()
