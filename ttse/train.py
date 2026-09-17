"""Two-stage training.

Stage 1: backbone + speaker encoder, static enrollment conditioning,
         standard utterance-level SI-SDR loss on Phase-I-style mixtures.
Stage 2: freeze backbone (optionally unfreeze late), train state module
         (and gates) on streaming scenarios with BPTT through state updates.

Loss (stage 2), per sequence, two terms only:
    L = -SI-SNR(est, target)   over the concatenated target-active chunks
      + w_sup * 10*log10(mean output power over the target-absent chunks)

Usage:
    python -m ttse.train --stage 1 --data <librispeech_dir> --out runs/stage1
    python -m ttse.train --stage 2 --state ttt --init runs/stage1/best.pt ...
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models.system import StreamingTSE
from .data.streaming_mix import StreamingScenario, librispeech_index
from .eval.metrics import si_snr


def masked_neg_sisnr(est, ref, active_mask, chunk):
    """-SI-SNR computed over the concatenation of active chunks only."""
    B = est.shape[0]
    losses = []
    for b in range(B):
        idx = active_mask[b].bool()
        if idx.sum() < 2:
            continue
        segs_e = est[b].unfold(0, chunk, chunk)[idx[: est.shape[1] // chunk]]
        segs_r = ref[b].unfold(0, chunk, chunk)[idx[: ref.shape[1] // chunk]]
        losses.append(-si_snr(segs_e.reshape(-1), segs_r.reshape(-1)))
    if not losses:
        return est.sum() * 0.0
    return torch.stack(losses).mean()


def absence_suppression_loss(est, active_mask, chunk):
    B = est.shape[0]
    losses = []
    for b in range(B):
        idx = (~active_mask[b].bool())[: est.shape[1] // chunk]
        if idx.sum() < 1:
            continue
        segs = est[b].unfold(0, chunk, chunk)[idx]
        losses.append(10 * torch.log10((segs ** 2).mean() + 1e-8))
    if not losses:
        return est.sum() * 0.0
    return torch.stack(losses).mean()


def collate(batch):
    T = max(x["mix"].shape[-1] for x in batch)
    C = max(x["active"].shape[-1] for x in batch)

    def pad(k, n):
        return torch.stack([F.pad(x[k], (0, n - x[k].shape[-1])) for x in batch])

    return {
        "mix": pad("mix", T), "target": pad("target", T), "masker": pad("masker", T),
        "enroll": torch.stack([x["enroll"] for x in batch]),
        "active": pad("active", C),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--data", required=True, help="LibriSpeech split dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None, choices=["static", "afw", "ttt"])
    ap.add_argument("--state-kw", default="{}", help="json dict for state module")
    ap.add_argument("--init", default=None, help="stage-1 checkpoint for stage 2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--w-sup", type=float, default=0.1)
    ap.add_argument("--chan-aug", action="store_true",
                    help="stage 2: random channel shift on in-mix target (sev 0-2)")
    ap.add_argument("--chan-aug3", action="store_true",
                    help="stage 2: channel shift aug including severity 3")
    args = ap.parse_args()
    args.state = args.state or ("static" if args.stage == 1 else "afw")
    if (args.stage == 1 and args.state != "static") or (args.stage == 2 and args.state == "static"):
        ap.error("stage 1 requires static; stage 2 requires afw/ttt")

    import json
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = StreamingTSE(state_name=args.state, state_kw=json.loads(args.state_kw)).to(dev)

    if args.stage == 1:
        # short two-speaker fully-overlapped mixtures, static conditioning.
        # The augmentation flags are accepted here too, so that a backbone can
        # be trained on the same channel shifts the learned states see: that
        # gives static enrollment the exposure a heuristic state cannot get
        # from stage 2, since it has nothing to train.
        ds = StreamingScenario(librispeech_index(args.data),
                               n_examples=args.steps_per_epoch * args.batch,
                               seed=args.seed, len_a=4.0, len_c=0.0,
                               absence_choices=(0.0,), tmr_range=(-5, 5),
                               channel_shift=-2 if args.chan_aug3 else (-1 if args.chan_aug else 0))
        lr = args.lr or 1e-3
        params = model.parameters()
    else:
        assert args.init, "stage 2 requires --init stage1 checkpoint"
        sd = torch.load(args.init, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded stage1: missing={len(missing)} unexpected={len(unexpected)}")
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.spk_enc.parameters():
            p.requires_grad = False
        ds = StreamingScenario(librispeech_index(args.data),
                               n_examples=args.steps_per_epoch * args.batch,
                               seed=args.seed, len_a=5.0, len_c=8.0,
                               absence_choices=(0.0, 1.0, 2.0, 4.0),
                               tmr_range=(-5, 5),
                               channel_shift=-2 if args.chan_aug3 else (-1 if args.chan_aug else 0))
        lr = args.lr or 3e-4
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        print(f"stage2 trainable params: {n_trainable}")
        if n_trainable == 0:  # static/ema have nothing to train
            print("state module has no trainable params — nothing to do")
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "best.pt"))
            return

    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                    collate_fn=collate, pin_memory=True, persistent_workers=args.num_workers > 0)

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, b in enumerate(dl):
            mix = b["mix"].to(dev, non_blocking=True)
            tgt = b["target"].to(dev, non_blocking=True)
            enr = b["enroll"].to(dev, non_blocking=True)
            act = b["active"].to(dev, non_blocking=True)

            if args.stage == 1:
                est = model.forward_static(mix, enr)
                loss = -si_snr(est, tgt).mean()
            else:
                est = model.forward_streaming(mix, enr, oracle_active=act)
                loss = masked_neg_sisnr(est, tgt, act, model.chunk)
                loss = loss + args.w_sup * absence_suppression_loss(est, act, model.chunk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item(); n += 1
            if i % 100 == 0:
                print(f"ep{ep} it{i} loss={tot / max(n, 1):.2f} "
                      f"({(time.time() - t0) / max(n, 1):.2f}s/it)", flush=True)
        sched.step()
        avg = tot / max(n, 1)
        torch.save({"model": model.state_dict(), "args": vars(args), "ep": ep},
                   os.path.join(args.out, "last.pt"))
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "args": vars(args), "ep": ep},
                       os.path.join(args.out, "best.pt"))
        print(f"=== epoch {ep} done, loss {avg:.3f} (best {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
