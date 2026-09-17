"""Causal Conv-TasNet-style TSE backbone.

Encoder -> stacked causal TCN separator (FiLM-conditioned on speaker state) -> decoder.
The backbone is agnostic to how the conditioning vector s_t is maintained;
state modules live in states.py and are swapped without touching this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv with left-only padding (strictly causal)."""

    def __init__(self, in_ch, out_ch, kernel, dilation=1, groups=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, groups=groups)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CumulativeLayerNorm(nn.Module):
    """cLN from Conv-TasNet: normalizes with cumulative statistics over time."""

    def __init__(self, ch, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, ch, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        step_sum = x.sum(1)                       # (B, T)
        step_sq = (x ** 2).sum(1)                 # (B, T)
        cum_sum = torch.cumsum(step_sum, dim=1)   # (B, T)
        cum_sq = torch.cumsum(step_sq, dim=1)
        cnt = torch.arange(1, T + 1, device=x.device, dtype=x.dtype) * C
        mean = cum_sum / cnt                      # (B, T)
        var = cum_sq / cnt - mean ** 2
        x = (x - mean.unsqueeze(1)) / (var.clamp_min(0).unsqueeze(1) + self.eps).sqrt()
        return x * self.gamma + self.beta


class FiLM(nn.Module):
    """Feature-wise linear modulation of separator features by speaker state."""

    def __init__(self, state_dim, ch):
        super().__init__()
        self.to_gamma = nn.Linear(state_dim, ch)
        self.to_beta = nn.Linear(state_dim, ch)

    def forward(self, x, s):
        # x: (B, C, T); s: (B, D) or (B, D, T) for time-varying state
        if s.dim() == 2:
            g = self.to_gamma(s).unsqueeze(-1)
            b = self.to_beta(s).unsqueeze(-1)
        else:
            g = self.to_gamma(s.transpose(1, 2)).transpose(1, 2)
            b = self.to_beta(s.transpose(1, 2)).transpose(1, 2)
        return x * (1 + g) + b


class TCNBlock(nn.Module):
    def __init__(self, ch, hid, kernel, dilation, state_dim):
        super().__init__()
        self.in_conv = nn.Conv1d(ch, hid, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = CumulativeLayerNorm(hid)
        self.dconv = CausalConv1d(hid, hid, kernel, dilation, groups=hid)
        self.prelu2 = nn.PReLU()
        self.norm2 = CumulativeLayerNorm(hid)
        self.out_conv = nn.Conv1d(hid, ch, 1)
        self.film = FiLM(state_dim, hid)

    def forward(self, x, s):
        y = self.norm1(self.prelu1(self.in_conv(x)))
        y = self.film(y, s)
        y = self.norm2(self.prelu2(self.dconv(y)))
        return x + self.out_conv(y)


class CausalTSE(nn.Module):
    """Causal TSE: mask estimation conditioned on a speaker state sequence.

    forward() takes the mixture and a state tensor s that is either
    (B, D) static or (B, D, T_frames) time-varying, so every state module
    (static / EMA / TTT / ...) plugs in identically.
    """

    def __init__(self, n_filters=256, win=32, stride=16, bn_ch=128,
                 hid_ch=256, kernel=3, n_blocks=7, n_repeats=3, state_dim=128):
        super().__init__()
        self.win, self.stride = win, stride
        self.encoder = nn.Conv1d(1, n_filters, win, stride=stride, bias=False)
        self.enc_norm = CumulativeLayerNorm(n_filters)
        self.bottleneck = nn.Conv1d(n_filters, bn_ch, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(bn_ch, hid_ch, kernel, 2 ** b, state_dim)
            for _ in range(n_repeats) for b in range(n_blocks)
        ])
        self.mask_conv = nn.Conv1d(bn_ch, n_filters, 1)
        self.decoder = nn.ConvTranspose1d(n_filters, 1, win, stride=stride, bias=False)
        self.state_dim = state_dim

    @property
    def frames_per_second(self):
        return 16000 // self.stride

    def encode(self, mix):
        # mix: (B, T_samples) -> (B, F, T_frames)
        w = F.relu(self.encoder(mix.unsqueeze(1)))
        return w

    def forward(self, mix, s):
        w = self.encode(mix)
        x = self.bottleneck(self.enc_norm(w))
        for blk in self.blocks:
            x = blk(x, s)
        m = torch.sigmoid(self.mask_conv(x))
        est = self.decoder(w * m).squeeze(1)
        # match input length
        if est.shape[-1] < mix.shape[-1]:
            est = F.pad(est, (0, mix.shape[-1] - est.shape[-1]))
        return est[..., :mix.shape[-1]]
