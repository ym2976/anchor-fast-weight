"""Load model weights while rejecting incomplete or incompatible checkpoints."""
import json
import torch
from .models.system import StreamingTSE


def load_model(path, state=None, state_kw=None, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    args = ck.get("args", {})
    name = state or args.get("state", "static")
    kw = state_kw if state_kw is not None else args.get("state_kw", {})
    if isinstance(kw, str):
        kw = json.loads(kw or "{}")
    model = StreamingTSE(state_name=name, state_kw=kw)
    weights = ck["model"]
    # Parameter-free frontier rules reuse only the trained separator/encoder.
    if name in {"static", "ema", "gated_ema", "vad_gated"}:
        weights = {k: v for k, v in weights.items() if not k.startswith("state.")}
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()
