"""Speaker state modules — the only thing that varies between systems.

Interface:
    state = module.init(e_enroll)            # per-sequence init from enrollment
    state, s = module.step(state, e_t, aux)  # consume chunk evidence, emit conditioning s_t

All modules emit s_t of the same dim so the backbone is shared verbatim.
`aux` carries cheap confidence features: chunk output energy (dB, roughly
normalized), cosine(e_t, previous readout). Oracle variants are handled by
feeding oracle evidence upstream, not by special-casing here.

Systems:
  S1 StaticState        s_t = e_enroll
  S2 EMAState           unconditional EMA, alpha fixed (swept at inference)
  S3 GatedEMAState      MoMuSE-like: hard confidence threshold gates the write
  S4 VADGatedState      static embedding + (oracle/estimated) target-activity gate over EMA
  S5 TTTState           proposed: rank-1 fast-weights memory with learned write/forget gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerState(nn.Module):
    def init(self, e_enroll):
        raise NotImplementedError

    def step(self, state, e_t, aux):
        raise NotImplementedError


# ---------------------------------------------------------------- S1
class StaticState(SpeakerState):
    def init(self, e_enroll):
        return {"s": e_enroll}

    def step(self, state, e_t, aux):
        return state, state["s"]


# ---------------------------------------------------------------- S2
class EMAState(SpeakerState):
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha

    def init(self, e_enroll):
        return {"s": e_enroll}

    def step(self, state, e_t, aux):
        s = self.alpha * state["s"] + (1 - self.alpha) * e_t
        s = F.normalize(s, dim=-1)
        return {"s": s}, s


# ---------------------------------------------------------------- S3
class GatedEMAState(SpeakerState):
    """MoMuSE-like: write only when confidence exceeds a threshold.

    Confidence = cos(e_t, current state). Threshold theta swept at inference.
    """

    def __init__(self, alpha=0.9, theta=0.5):
        super().__init__()
        self.alpha = alpha
        self.theta = theta

    def init(self, e_enroll):
        return {"s": e_enroll}

    def step(self, state, e_t, aux):
        conf = F.cosine_similarity(e_t, state["s"], dim=-1, eps=1e-8)  # (B,)
        gate = (conf > self.theta).float().unsqueeze(-1)
        s = gate * (self.alpha * state["s"] + (1 - self.alpha) * e_t) + (1 - gate) * state["s"]
        s = F.normalize(s, dim=-1)
        return {"s": s}, s


# ---------------------------------------------------------------- S4
class VADGatedState(SpeakerState):
    """Static + activity gate: EMA update only when target is active.

    Activity signal comes in aux["active"] (B,) in {0,1}: oracle from the
    simulation script, or estimated from output energy at inference.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def init(self, e_enroll):
        return {"s": e_enroll}

    def step(self, state, e_t, aux):
        act = aux["active"].float().unsqueeze(-1)
        s = act * (self.alpha * state["s"] + (1 - self.alpha) * e_t) + (1 - act) * state["s"]
        s = F.normalize(s, dim=-1)
        return {"s": s}, s


# ---------------------------------------------------------------- S5 (proposed)
class AnchoredFastWeightState(SpeakerState):
    """Fast-weights speaker memory with learned write/forget gates.

    State is W in R^{d x d}. Per chunk (closed-form, no autograd at inference):

        k_t, v_t = K e_t, V e_t
        err_t    = W_{t-1} k_t - v_t
        W_t      = (1 - a_t) W_{t-1} - eta_t * err_t k_t^T     (rank-1 write)
        s_t      = normalize(W_t q + e_enroll)                 (residual readout)

    eta_t (write rate) and a_t (forget rate) are produced by a small gate net
    from [e_t, cos(e_t, prev readout), energy] — learned end-to-end in stage 2.
    This is the entire novelty: WHAT to write is a gradient step, WHETHER to
    write is learned, and the enrollment anchor is a residual so the memory
    only ever has to model the *deviation* from enrollment.
    """

    def __init__(self, dim=128, gate_hidden=64, eta_max=1.0, alpha_max=0.1,
                 ):
        super().__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.q = nn.Parameter(torch.randn(dim) / dim ** 0.5)
        self.gate = nn.Sequential(
            nn.Linear(dim + 2, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.eta_max = eta_max
        self.alpha_max = alpha_max
        # init gate biases so training starts near "write a little, forget nothing"
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([-2.0, -4.0]))

    def init(self, e_enroll):
        B, d = e_enroll.shape
        W = torch.zeros(B, d, d, device=e_enroll.device, dtype=e_enroll.dtype)
        return {"W": W, "anchor": e_enroll, "prev_s": e_enroll}

    def step(self, state, e_t, aux):
        W, anchor = state["W"], state["anchor"]
        conf = F.cosine_similarity(e_t, state["prev_s"], dim=-1, eps=1e-8)
        feats = torch.cat([e_t, conf.unsqueeze(-1), aux["energy"].unsqueeze(-1)], dim=-1)
        g = self.gate(feats)
        eta = self.eta_max * torch.sigmoid(g[:, 0:1])
        alpha = self.alpha_max * torch.sigmoid(g[:, 1:2])

        lam_t = alpha.clamp(max=0.99)
        k = self.K(e_t)                                    # (B,d)
        v = self.V(e_t)
        err = torch.einsum("bij,bj->bi", W, k) - v         # (B,d)
        # rank-1 write: W <- (1-a) W - eta * err k^T
        W = (1 - lam_t.unsqueeze(-1)) * W - (eta * err).unsqueeze(-1) * k.unsqueeze(1)

        r = torch.einsum("bij,j->bi", W, self.q)           # readout
        s = F.normalize(anchor + r, dim=-1)
        return {"W": W, "anchor": anchor, "prev_s": s}, s


# Historical checkpoint/API name retained for compatibility.
TTTState = AnchoredFastWeightState


def build_state(name, **kw):
    table = {"static": StaticState, "ema": EMAState,
             "gated_ema": GatedEMAState, "vad_gated": VADGatedState,
             "afw": AnchoredFastWeightState, "ttt": AnchoredFastWeightState}
    if name not in table:
        raise ValueError(f"Unknown state {name!r}; choose from {tuple(table)}")
    return table[name](**kw)
