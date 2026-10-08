# Music Generation Using Char-RNN

A character-level RNN that learns folk melodies written in **abc notation**
and composes new, original, playable tunes — character by character.

Theory 22PC1DS402 · Lab 22PC2DS402 · Deep Learning Applications ·
B.Tech Data Science (Final Year)

> Every number in this README is measured in this build and traces to
> `music-generation-char-rnn/outputs/results.json`,
> `music-generation-char-rnn/outputs/experiments.csv`, or a log/figure in
> `music-generation-char-rnn/outputs/`. Nothing is copied from blogs or papers.

## Problem

**Given** a seed/start string in abc notation (a valid header plus an opening
fragment), **generate** a continuation that forms a complete, new folk-style
tune: valid abc headers, consistent meter and key, well-formed bar lines, a
clean ending, not a copy of any training tune, convertible to MIDI and
playable as audio.

Music-as-text turns composition into **next-character prediction** — a
multiclass classification problem at every step (Softmax + cross-entropy)
— with an output a listener can judge in seconds.

## Dataset — Nottingham Music Database

Public-domain traditional folk tunes (reels, jigs, hornpipes, waltzes) in
plain-text abc (~5 MB). Used the cleaned mirror
`github.com/jukedeck/nottingham-dataset` (`ABC_cleaned/`, 14 files),
vendored under `music-generation-char-rnn/data/raw/`. No other data used.

| Statistic (measured) | Value |
|---|---|
| Tunes total | 1034 (regex `^X:` count) |
| Train / validation | 930 / 104 tunes — split **by tune** (seed 42), never by windows (avoids leakage) |
| Total characters | 390475 |
| Vocabulary | 86 chars (tens, as expected — cleaning verified) |
| Dropped (no `K:` line) | 0 |
| Train / val windows (seq len 64) | 287862 / 34370 (stride 1) |
| Mean tune length | 375.6 chars |
| Meters | 4/4: 548, 6/8: 360, 3/4: 60, 2/4: 41, 9/8: 13, 2/2: 7, 3/2: 3, 6/4: 2 |
| Keys | G: 361, D: 358, A: 125, C: 54, Am: 39, Em: 29, F: 25, … |

EDA figures: `music-generation-char-rnn/outputs/figures/` (`char_dist`,
`meter_dist`, `key_dist`, `tune_lengths`).

## Architecture

Main model — many-to-one Char-RNN (`src/model.py`, 1,623,126 params):

- Input: integer char IDs, shape (batch, 64), from `vocab.json` (no one-hot;
  the learned Embedding replaces it — this is **not** Word2Vec)
- Embedding 86×256 → LSTM 256 → LSTM 256 → LSTM 256 → Dropout 0.3 →
  Dense(86) + Softmax (next-character distribution)
- Loss: cross-entropy · Metrics: accuracy, perplexity = exp(loss), per epoch
- Optimizer: Adam, lr 3e-3, gradient clipping norm 1.0

Variants (same data/split): **many-to-many** (per-step predictions, Lab 13–14
arch.), **stateful** (ordered batches, hidden carried across batches, reset
each epoch), **Char-CNN** (Emb + Conv1D k5/k3 ReLU + global max-pool, 634,710
params). Baselines: most-frequent char, char 4-gram (+add-1 smoothing), small
Elman RNN (41,430 params).

Framework note: TF 2.11+ has no native-Windows GPU, so the identical maths
was built in PyTorch 2.1+cu121 to use the RTX 3050 (E12 measures ~14× vs CPU).

## Training

| | Smoke (pipeline proof) | Full (reported result) |
|---|---|---|
| Data | 15% tunes | 100% (144,162 windows, stride 2 — laptop-GPU budget, logged) |
| Epochs | 5 | max 60, early stop patience 8 → **stopped ep 19** (best val ep 11) |
| Batch / time | 64 / ~20 s·epoch⁻¹ | 128 / ~40 s·epoch⁻¹ (~13 min total, RTX 3050) |
| Val result | acc 0.585, ppl 3.7 | **loss 1.049, acc 0.675, ppl 2.85** (train 0.635/0.783/1.89) |

Best-on-val-loss checkpoint: `music-generation-char-rnn/outputs/checkpoints/main_full.pt`.
Curves: `music-generation-char-rnn/outputs/figures/curves_main_full.png`
(plotted from saved history). The main model beats all three baselines
(0.675 > 0.543 > 0.521 > 0.155).

## Results — baselines

| Baseline | Val acc | Val ppl |
|---|---|---|
| Most-frequent char | 0.155 | 2091.1 |
| Char 4-gram | 0.521 | 6.1 |
| Small Elman RNN | 0.543 | 4.6 |
| **Main Char-RNN** | **0.675** | **2.85** |

## Results — experiments E1–E12 (reduced budget: 15% data / 8 epochs)

Full table: `music-generation-char-rnn/outputs/experiments.csv`. One variable
changed per run.

| ID | Variable | Outcome (val acc) | Takeaway |
|---|---|---|---|
| E1 | Optimizer | RMSProp **0.607**, Adam 0.574, AdaGrad 0.548, SGD+Mom 0.340, NAG 0.321, SGD/Adadelta 0.155 | Adaptive methods win on a short budget |
| E2 | Clip on/off, lr=1e-2 | 0.155 / 0.155 (both collapse) | Clipping can't rescue a 3×-too-high LR |
| E3 | Dropout 0 / 0.3 / 0.5 | 0.569 / 0.572 / 0.570 | Indistinguishable at 8 epochs; gap opens by ep 19 |
| E4 | +LayerNorm | **0.608** vs 0.572 | Small win, faster early convergence |
| E5 | Activation/init | analysis (labeled) | E2 + CNN-ReLU + E4 jointly: scale control matters most |
| E6 | LSTM 128/256/512 | 0.555 / 0.572 / 0.575 (0.49M/1.62M/5.85M params, 6/9/22 s·ep⁻¹) | 256 is the knee — steep cost, flat gain |
| E7 | Seq len 32/64/100 | **0.585** / 0.572 / 0.566 | Shorter context learns local patterns faster |
| E8 | Temperature 0.6–1.2 | 6/6 valid at every temp (24 attempts) | Validity holds; musical focus loosens with T |
| E9 | Stateful vs stateless | 0.549 vs 0.572 | Shuffling wins on tune-sliced data |
| E10 | RNN vs CNN vs many2many | 0.675 / 0.510 / 0.587 | Recurrence beats local convolution here |
| E11 | LR × batch (3×2) | best **1e-3 × 32 (0.583)**; 1e-2 diverges (0.448/0.123) | — |
| E12 | CPU vs GPU, same config | 54.4 vs 3.9 s·epoch⁻¹ (**~13.9×**) | Unit IV evidence |

## Results — generation

Autoregressive sampling (real header + fragment seed, stop at corpus-style
`:|`/`|` ending), 6 attempts × 4 temperatures = **24 attempts**:

- **Validity 24/24 (1.00)** — raw 19/24, post-repair 5/24; only documented
  repairs (header prepend, final-bar append, dangling-opener drop)
- Validator rule is corpus-measured: repeats valid iff openers ≤ closers
  (584/930 train tunes have *more* closers — repeat-from-start convention)
- **Originality 28/28** (exact-match vs training tunes)
- **28 triplets** in `music-generation-char-rnn/outputs/generated/` (24 main
  + 3 variant samples + tabla sketch), each `.abc + .mid + .wav`, WAVs
  asserted non-silent. Featured: `gen_T0.6_0` (G/4/4), `gen_T0.8_0` (D/4/4),
  `gen_T1.0_0` (D/6/8), `gen_T1.2_0` (G/4/4), `gen_tabla_01`
  (rhythm-conditioned extension demo — abc has no timbre/bols, so this is
  explicitly **not** a tabla model)

## How to run

```bat
cd music-generation-char-rnn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data_prep.py --split-seed 42
python src/train.py --config smoke
python src/train.py --config full --tag main_full
python src/generate.py --checkpoint outputs/checkpoints/main_full.pt --temperatures 0.6 0.8 1.0 1.2 --attempts-per-temp 6
python src/convert.py --in outputs/generated --out outputs/generated
python src/evaluate.py --all
```

Needs Python 3.11 + NVIDIA GPU (PyTorch CUDA build, tested on RTX 3050);
CPU-only works, ~14× slower. Experiment runs, e.g.
`python src/train.py --config exp --tag E1_adam --opt adam --epochs 8 --subset 0.15`
(full list: `experiments/run_all.py`). Commands tested verbatim from a clean
checkout (smoke scale; logged in `outputs/phase_log.md`).

## Layout

```
music-generation-char-rnn/
  data/raw/  data/processed/  (corpus.txt, train/val.txt, vocab.json, sequences_meta.json)
  src/  data_prep.py  model.py  train.py  generate.py  convert.py  evaluate.py  common.py
  experiments/  run_all.py + per-topic notes
  outputs/  results.json  experiments.csv  phase_log.md  logs/  figures/  generated/  checkpoints/
  report/  report.md (full report + viva sheet)  failure_gallery.abc/.txt
```

## Limitations

Stride 2 (not 1) on the full run — laptop-GPU budget; mild overfitting
(train 0.78 vs val 0.67) on ~1k tunes; sine-synth WAVs (no soundfont);
single melody line. Details + conclusion in `report/report.md` Ch.7.

## Citation

Nottingham Music Database (public-domain folk tunes), cleaned version via
`github.com/jukedeck/nottingham-dataset` (Jukedeck).
