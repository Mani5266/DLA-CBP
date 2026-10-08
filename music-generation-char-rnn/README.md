# Music Generation Using Char-RNN

Character-level RNN that learns folk melodies in **abc notation** from the
Nottingham Music Database (public domain, 1034 tunes) and composes new tunes.
Theory 22PC1DS402 · Lab 22PC2DS402 · Deep Learning Applications.

## 1. Setup (local, Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Needs Python 3.11 + NVIDIA GPU (PyTorch CUDA build, tested on RTX 3050).
CPU-only works too (slower; see E12 timings in `outputs/results.json`).
Why PyTorch and not TensorFlow: TF 2.11+ has no native-Windows GPU support
(last GPU build was TF 2.10, Python ≤ 3.10) — the model maths is identical
(Embedding + LSTM + Softmax + cross-entropy). Recorded in
`outputs/phase_log.md` and the report.

## 2. Data

```bat
python src/data_prep.py --split-seed 42
```

Downloads are already vendored under `data/raw/nottingham-dataset/`
(Jukedeck cleaned mirror, public domain). Produces `data/processed/`:
`corpus.txt`, `train.txt` / `val.txt` (tune-level 90/10 split, seed 42),
`vocab.json`, `sequences_meta.json`.

## 3. Train — smoke first, then full (GPU)

```bat
python src/train.py --config smoke
python src/train.py --config full --tag main_full
```

Smoke: 15% data, 5 epochs (~2 min, proves the pipeline end-to-end).
Full: all data, up to 60 epochs with early stopping (patience 8) on
validation loss; best checkpoint -> `outputs/checkpoints/main_full.pt`,
history -> `outputs/logs/history_main_full.json`.

## 4. Generate + convert (at least 5 tunes, 4 temperatures)

```bat
python src/generate.py --checkpoint outputs/checkpoints/main_full.pt --temperatures 0.6 0.8 1.0 1.2 --attempts-per-temp 6
python src/convert.py --in outputs/generated --out outputs/generated
python src/evaluate.py --all
```

`generate.py` samples character-by-character with temperature, validates
every tune (X:/T:/M:/K: order, bar lines, clean ending, balanced repeats,
in-vocabulary), applies only the documented repairs, and records raw +
post-repair validity in `outputs/results.json`. `convert.py` renders
`.abc -> .mid -> .wav` (pure-Python MIDI + sine-synth render, no soundfont
needed). `evaluate.py` updates `outputs/results.json` + `outputs/experiments.csv`.

## 5. Experiments E1–E12

```bat
python src/train.py --config exp --tag E1_adam --opt adam --epochs 8 --subset 0.15
```

One variable per run; reduced budget (15% data / 8 epochs) except the main
model. E8/E9/E10 reuse Phase 5–7 runs. Full table: `outputs/experiments.csv`.

## Layout

```
data/raw/  data/processed/
src/  data_prep.py  model.py  train.py  generate.py  evaluate.py  convert.py
experiments/  (run tags + notes; configs live in train.py PRESETS)
outputs/  results.json  phase_log.md  logs/  figures/  generated/  checkpoints/
report/  README.md
```
