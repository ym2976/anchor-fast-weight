"""Streaming TSE system: backbone + speaker encoder + pluggable state module.

Chunk loop (hop = chunk_ms) used identically for stage-2 training (BPTT
through state updates) and inference. Chunk t is separated with s_{t-1};
its output produces evidence e_t which updates the state for chunk t+1.

Memory note: with BPTT over N chunks, holding backbone activations for every
chunk step OOMs quickly. The backbone is frozen in stage 2, so we wrap its
per-chunk forward in gradient checkpointing (recompute in backward) - memory
becomes O(N * chunk_io) instead of O(N * full_activations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .backbone import CausalTSE
from .speaker_encoder import SpeakerEncoder
from .states import build_state


class StreamingTSE(nn.Module):
    def __init__(self, state_name="afw", state_kw=None, chunk_ms=250, sr=16000,
                 backbone_kw=None, spk_kw=None, ctx_chunks=16):
        super().__init__()
        self.backbone = CausalTSE(**(backbone_kw or {}))
        self.spk_enc = SpeakerEncoder(**(spk_kw or {}))
        self.state = build_state(state_name, **(state_kw or {}))
        self.chunk = int(sr * chunk_ms / 1000)
        self.sr = sr
        self.ctx_chunks = ctx_chunks            # left-context budget per chunk step

    # ---------------- offline (stage 1): static conditioning, full utterance
    def forward_static(self, mix, enroll):
        e = self.spk_enc(enroll)
        return self.backbone(mix, e)

    # ---------------- streaming: chunked with state updates
    def forward_streaming(self, mix, enroll, oracle_active=None,
                          return_trace=False):
        """mix: (B, T); enroll: (B, T_e).
        oracle_active: (B, n_chunks) 0/1 target activity per chunk (for VAD-gated).
        """
        B, T = mix.shape
        if T < 32 or enroll.shape[-1] < 32:
            raise ValueError("Mixture and enrollment must have at least 32 samples")
        e0 = self.spk_enc(enroll)
        st = self.state.init(e0)
        n_chunks = (T + self.chunk - 1) // self.chunk
        use_ckpt = self.training and torch.is_grad_enabled()

        outs, trace_s = [], []
        s = e0
        for t in range(n_chunks):
            lo, hi = t * self.chunk, min((t + 1) * self.chunk, T)
            # separate with *current* state; backbone is causal so feeding a
            # bounded left context and slicing the tail approximates streaming
            # Cumulative normalization is recomputed on this bounded context.
            ctx_lo = max(0, hi - self.chunk * self.ctx_chunks)
            seg = mix[:, ctx_lo:hi]
            if use_ckpt:
                est_full = checkpoint(self.backbone, seg, s, use_reentrant=False)
            else:
                est_full = self.backbone(seg, s)
            est = est_full[:, lo - ctx_lo:]
            outs.append(est)

            # Closed-loop evidence from the extracted chunk.
            src = F.pad(est, (0, max(0, 32 - est.shape[-1])))
            if use_ckpt:
                e_t = checkpoint(self.spk_enc.chunk_embedding, src, use_reentrant=False)
            else:
                e_t = self.spk_enc.chunk_embedding(src)
            energy = 10 * torch.log10((src ** 2).mean(-1) + 1e-8) / 40.0  # rough norm
            aux = {"energy": energy.clamp(-2, 2)}
            if oracle_active is not None:
                aux["active"] = oracle_active[:, t]
            else:
                aux["active"] = (energy > -1.0).float()  # estimated VAD fallback
            st, s = self.state.step(st, e_t, aux)
            if return_trace:
                trace_s.append(s.detach())

        est = torch.cat(outs, dim=-1)[:, :T]
        if return_trace:
            return est, {"s": torch.stack(trace_s, dim=1)}  # (B, n_chunks, D)
        return est
