# Anchored Fast Weights (AFW)

A PyTorch implementation of **anchored fast-weight memory for streaming target speaker extraction**, with the **22 heuristic settings** used to measure the stability–plasticity frontier.

This release contains the model, two-stage training, audio inference, scenario construction, and frontier evaluation. Ablations, architecture controls, private cluster scripts, and manuscript assets are excluded. **Pretrained checkpoints and datasets are not included.** The release has CPU correctness/smoke tests; it has not been retrained to reproduce paper scores.

## Model

A Conv-TasNet-style separator uses FiLM conditioning from a speaker state. A shared speaker encoder embeds enrollment audio and each extracted chunk. The AFW state has 41,410 trainable parameters at the default dimension of 128.

For chunk t, separation uses the previous state. The extracted audio supplies evidence e_t for the next chunk:

```text
k_t = K e_t; v_t = V e_t
(eta_t, alpha_t) = bounded sigmoid gates(e_t, cosine(e_t, s_prev), energy)
W_t = (1 - alpha_t) W_prev - eta_t (W_prev k_t - v_t) k_t^T
s_t = normalize(enrollment + W_t q)
```

W starts at zero. The enrollment anchor stays fixed. Defaults are eta_max=1 and alpha_max=0.1. Stage 2 differentiates through the closed loop; inference uses tensor updates without backward/optimizer steps. `afw` is the public state name; `ttt` and `TTTState` remain compatible aliases for original checkpoints.

Audio is mono, 16 kHz. Chunks are 250 ms, with at most 16 chunks of recomputed context (4 seconds including the current chunk). This is a chunked research implementation, not a cached sample-by-sample streaming engine. Cumulative normalization restarts on each context window, so bounded-context output is not claimed to equal full-utterance output.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

Python >=3.10, PyTorch >=2.1, NumPy, and SoundFile are required. Install a PyTorch build suitable for your accelerator. Tests were run locally on CPU with Python 3.11 and PyTorch 2.13.0.

## Train

Provide LibriSpeech FLAC files in the standard `split/speaker/chapter/*.flac` layout. Use disjoint training and test splits. At least two speakers with three recordings each are required; enrollment uses a different recording from the target mixture. Datasets retain their own terms and are not distributed here.

```bash
python -m ttse.train --stage 1 \
  --data /path/to/LibriSpeech/train-clean-360 --out runs/stage1 \
  --epochs 60 --batch 16 --steps-per-epoch 2000

python -m ttse.train --stage 2 --state afw --chan-aug3 \
  --init runs/stage1/best.pt \
  --data /path/to/LibriSpeech/train-clean-360 --out runs/afw_seed0 \
  --epochs 20 --batch 8 --steps-per-epoch 1000 --seed 0
```

Stage 1 trains the separator and speaker encoder with static enrollment. Stage 2 freezes both and trains AFW with active-region negative SI-SNR plus absent-region output-power suppression (weight 0.1). Training absences are 0/1/2/4 seconds; TMR ranges from -5 to 5 dB. `--chan-aug3` samples channel severity 0/1/2/3. `best.pt` means lowest **training loss**, not held-out validation selection. Adjust batch size/workers for your machine. Seeds control scenario sampling, but exact cross-device reproducibility is not guaranteed.

## Infer

```bash
python -m ttse.infer --ckpt runs/afw_seed0/best.pt \
  --mix mixture.wav --enroll enrollment.wav --out extracted.wav --device cpu
```

The API is `StreamingTSE(state_name="afw").forward_streaming(mix, enroll)` for tensors shaped `[batch, samples]`. Each call initializes a fresh sequence state. Use a trained checkpoint for meaningful extraction; randomly initialized weights are only useful for smoke tests. Use checkpoints from a trusted source.

## The 22 frontier settings

The complete machine-readable list is [`ttse/frontier_configs.json`](ttse/frontier_configs.json).

| Family | Count | Swept values | Fixed value |
|---|---:|---|---|
| EMA | 9 | alpha: 0, .5, .8, .9, .95, .97, .99, .995, .999 | — |
| Confidence-gated EMA | 8 | theta: .2, .3, .4, .5, .6, .7, .8, .9 | alpha=.9 |
| Oracle-VAD EMA | 5 | alpha: .9, .95, .97, .99, .995 | clean-target activity |

All states are L2-normalized after updating. Confidence gating uses `cos(e_t, s_prev) > theta`. Oracle activity is derived from clean-target chunk mean-square power >1e-6; it is an experimental oracle, not an estimated VAD. All 22 settings reuse the **same stage-1 separator and speaker encoder**. Static enrollment is an extra reference, outside the 22.

```bash
python -m ttse.frontier --list
python -m ttse.frontier \
  --ckpt runs/stage1/best.pt --afw-ckpt runs/afw_seed0/best.pt \
  --data /path/to/LibriSpeech/test-clean \
  --out results/frontier --n 200 --seed 1234
```

Omit `--afw-ckpt` to run the heuristics and static reference only. The runner defaults to CUDA when available, otherwise CPU. It writes per-sequence JSONL, run arguments, and `summary.csv` with the nondominated heuristic points.

- **Plasticity (x):** severity-3 channel mismatch, no absence, TMR=0 dB.
- **Stability (y):** 30-second absence, severity=0, TMR=0 dB.
- Both axes measure phase-III SI-SNR improvement over the mixture. Phase I is 5 seconds; phase III is 8 seconds. Each setting sees identically seeded scenarios.

Channel transforms affect the target mixture speech, leaving enrollment unchanged. TMR scaling uses the full scenario, including target-absent samples, matching the original implementation. File enumeration is sorted in this release for stable sampling; older runs with unsorted enumeration may select different utterances even with the same seed. The runner computes results from checkpoints and audio; it does not embed paper figure coordinates. Full 22-setting evaluation is computationally expensive.

## Layout

```text
ttse/models/             separator, speaker encoder, AFW and heuristic states
ttse/data/               seeded mixtures and channel mismatch
ttse/train.py           two-stage training
ttse/infer.py           WAV inference
ttse/frontier.py        paired-axis evaluation of all 22 settings
ttse/frontier_configs.json
 tests/                 update, gating, and streaming gradient tests
```

MIT license. Dataset and dependency licenses are separate.
