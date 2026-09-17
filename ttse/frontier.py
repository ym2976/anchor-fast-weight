"""Evaluate the 22 frontier settings on identical seeded scenarios."""
import argparse
import csv
import json
from importlib.resources import files
from pathlib import Path
import torch
from .checkpoints import load_model
from .data.streaming_mix import StreamingScenario, librispeech_index
from .eval.metrics import si_snr_improvement


def configurations():
    return json.loads(files("ttse").joinpath("frontier_configs.json").read_text())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", help="Shared stage-1 checkpoint")
    ap.add_argument("--afw-ckpt", help="Optionally include trained AFW")
    ap.add_argument("--data", help="LibriSpeech test split")
    ap.add_argument("--out", default="results/frontier")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    configs = configurations()
    if args.list:
        print(json.dumps(configs, indent=2)); return
    if not args.ckpt or not args.data or args.n < 1:
        ap.error("--ckpt, --data, and --n >= 1 are required")
    index = librispeech_index(args.data)
    scenarios = {
        "plasticity": StreamingScenario(index, n_examples=args.n, seed=args.seed,
            absence_choices=(0.,), tmr_range=(0, 0), channel_shift=3),
        "stability": StreamingScenario(index, n_examples=args.n, seed=args.seed,
            absence_choices=(30.,), tmr_range=(0, 0), channel_shift=0),
    }
    configs = configs + [dict(id="static", state="static", state_kw={})]
    if args.afw_ckpt:
        configs.append(dict(id="afw", state="afw", state_kw=None))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    rows = []
    for cfg in configs:
        model = load_model(args.afw_ckpt if cfg["id"] == "afw" else args.ckpt,
                           cfg["state"], cfg["state_kw"], args.device)
        row = {"id": cfg["id"]}
        with (out / f'{cfg["id"]}.jsonl').open("w") as f:
            for axis, ds in scenarios.items():
                scores = []
                for i in range(len(ds)):
                    ex = ds[i]
                    mix, target, enroll = [ex[k].unsqueeze(0).to(args.device)
                                           for k in ("mix", "target", "enroll")]
                    active = ex["active"].unsqueeze(0).to(args.device) if cfg.get("oracle_activity") else None
                    est = model.forward_streaming(mix, enroll, oracle_active=active)
                    start = ex["bounds"][1]
                    score = si_snr_improvement(est[:, start:], mix[:, start:], target[:, start:]).item()
                    scores.append(score)
                    f.write(json.dumps(dict(axis=axis, i=i, sisnri_p3=score,
                        spk_a=ex["spk_a"], spk_b=ex["spk_b"])) + "\n")
                row[axis] = sum(scores) / len(scores)
        rows.append(row)
        print(row, flush=True)
    heuristics = rows[:22]
    for r in rows:
        r["heuristic_frontier"] = r in heuristics and not any(
            s["plasticity"] >= r["plasticity"] and s["stability"] >= r["stability"]
            and (s["plasticity"] > r["plasticity"] or s["stability"] > r["stability"])
            for s in heuristics)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
