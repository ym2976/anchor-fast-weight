"""Small causal speaker encoder E.

Used for (a) enrollment embedding and (b) per-chunk evidence e_t = E(y_hat_t).
Trained jointly with the backbone (stage 1). Deliberately NOT an off-the-shelf
SV model: evaluation SpkSim uses an external, different-family SV model
(speechbrain ECAPA) to avoid metric leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CausalConv1d, CumulativeLayerNorm


class SpeakerEncoder(nn.Module):
    def __init__(self, n_filters=256, win=32, stride=16, ch=128, emb_dim=128, n_blocks=4):
        super().__init__()
        self.encoder = nn.Conv1d(1, n_filters, win, stride=stride, bias=False)
        self.proj = nn.Conv1d(n_filters, ch, 1)
        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            self.blocks.append(nn.Sequential(
                CausalConv1d(ch, ch, kernel=3, dilation=2 ** b),
                nn.PReLU(),
                CumulativeLayerNorm(ch),
            ))
        self.head = nn.Conv1d(ch, emb_dim, 1)
        self.emb_dim = emb_dim

    def frame_features(self, wav):
        # wav: (B, T) -> (B, emb_dim, T_frames); strictly causal
        x = F.relu(self.encoder(wav.unsqueeze(1)))
        x = self.proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)

    def forward(self, wav):
        """Utterance-level embedding: mean-pool over time, L2-normalized."""
        f = self.frame_features(wav)
        e = f.mean(-1)
        return F.normalize(e, dim=-1)

    def chunk_embedding(self, wav):
        """Embedding of a (short) chunk — same pooling, kept separate for clarity."""
        return self.forward(wav)
