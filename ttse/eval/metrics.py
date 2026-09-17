"""Frozen evaluation metrics — written before any model results exist.

Active segments : SI-SNR(i)
Absence segments: output suppression (dB)  [SI-SNR undefined there — never use it]
Identity        : SpkSim / Speaker Confusion Rate over 1.6 s sliding windows,
                  computed with an EXTERNAL SV model (speechbrain ECAPA),
                  never with the jointly-trained encoder E.
Recovery        : T_reacq = time after target return to reach 90% of the
                  system's own post-return steady-state SI-SNR.
"""

import torch
import torch.nn.functional as F


def si_snr(est, ref, eps=1e-8):
    """Scale-invariant SNR in dB. est, ref: (..., T). Ref must be non-silent."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / ((ref ** 2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    return 10 * torch.log10((proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps) + eps)


def si_snr_improvement(est, mix, ref):
    return si_snr(est, ref) - si_snr(mix, ref)


def suppression_db(est, mix, eps=1e-8):
    """During target absence: how much output power is below mixture power.
    Positive = output is quieter than the (masker-only) mixture. Higher is better."""
    p_est = (est ** 2).mean(-1)
    p_mix = (mix ** 2).mean(-1)
    return 10 * torch.log10(p_mix / (p_est + eps) + eps)


@torch.no_grad()
def windowed_spk_sim(sv_model, est, ref_emb_a, ref_emb_b, sr=16000,
                     win_sec=1.6, hop_sec=0.4, energy_floor_db=-45.0):
    """Sliding-window speaker similarity of est vs. target A and masker B.

    sv_model: callable wav (B, T) -> L2-normalized emb (B, D)  [external ECAPA]
    Returns dict with per-window sims and the confusion mask (sim_b > sim_a),
    windows below the energy floor excluded (nothing to attribute).
    """
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    T = est.shape[-1]
    sims_a, sims_b, keep = [], [], []
    for lo in range(0, T - win + 1, hop):
        seg = est[..., lo:lo + win]
        e_db = 10 * torch.log10((seg ** 2).mean(-1) + 1e-10)
        emb = sv_model(seg)
        sims_a.append(F.cosine_similarity(emb, ref_emb_a, dim=-1))
        sims_b.append(F.cosine_similarity(emb, ref_emb_b, dim=-1))
        keep.append(e_db > energy_floor_db)
    sims_a = torch.stack(sims_a, -1)   # (B, n_win)
    sims_b = torch.stack(sims_b, -1)
    keep = torch.stack(keep, -1)
    confusion = (sims_b > sims_a) & keep
    return {"sim_a": sims_a, "sim_b": sims_b, "keep": keep,
            "confusion_rate": confusion.float().sum(-1) / keep.float().sum(-1).clamp(min=1)}


def reacquisition_time(est, ref, mix, return_sample, sr=16000,
                       win_sec=0.5, frac=0.9, tail_sec=3.0):
    """Threshold-free T_reacq for ONE sequence (1D tensors).

    Steady state = mean windowed SI-SNR over the last tail_sec of the sequence
    (the system's OWN post-return plateau). T_reacq = first time after return
    where windowed SI-SNR reaches frac * plateau (in linear-dB terms, we use
    plateau - (1-frac)*10 dB band edge instead when plateau is small).
    Returns seconds, or inf if never reached.
    """
    win = int(win_sec * sr)
    T = est.shape[-1]
    curve, times = [], []
    for lo in range(return_sample, T - win + 1, win // 2):
        seg_ref = ref[lo:lo + win]
        if (seg_ref ** 2).mean() < 1e-8:
            continue
        curve.append(si_snr(est[lo:lo + win], seg_ref).item())
        times.append(lo / sr)
    if len(curve) < 3:
        return float("inf")
    tail_n = max(1, int(tail_sec / (win_sec / 2)))
    plateau = sum(curve[-tail_n:]) / tail_n
    target_level = plateau - (1 - frac) * max(plateau, 10.0)
    for t, v in zip(times, curve):
        if v >= target_level:
            return t - return_sample / sr
    return float("inf")
