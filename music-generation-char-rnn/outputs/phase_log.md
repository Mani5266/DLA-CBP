# Phase log (PRD 1.1, 1.4) — one entry per phase with date, command, outcome.

## 2026-10-08 — Phase 0: environment + dataset — PASS
- Tree created per PRD 5.1 under `music-generation-char-rnn/`.
- `pip install mido` (only new dep; torch 2.1+cu121, numpy, matplotlib,
  scipy, pandas, sklearn pre-installed).
- Dataset: `git clone --depth 1 jukedeck/nottingham-dataset`
  -> `data/raw/nottingham-dataset/ABC_cleaned/` (14 files).
- Tune count (regex `^X:`): **1034 tunes** (matches PRD ~1033 cleaned).
- 3 tunes inspected: headers X/T/M/L/K present, bar symbols present.
- TF-Windows trap (PRD 5): confirmed applicable — TF 2.11+ has no native
  Windows GPU. Decision: PyTorch 2.1+cu121, `torch.cuda` = True
  (RTX 3050 Laptop GPU). Same Char-RNN maths (Emb+LSTM+CE); logged in
  `results.json.environment.note`. Local GPU training replaces Colab.
- Conversion smoke: `python src/convert.py --in outputs/phase0_test.abc
  --out outputs` -> 1/1 OK (phase0_test.mid 365 B, phase0_test.wav 1.1 MB).

## 2026-10-08 — Phase 1: EDA — PASS
- `python src/evaluate.py --eda` -> tunes=1034, chars=390475, vocab=86
  (small, as expected). Meters: 4/4:548, 6/8:360, 3/4:60, 2/4:41, 9/8:13.
- Figures: char_dist, meter_dist, key_dist, tune_lengths -> outputs/figures/.

## 2026-10-08 — Phase 2: data prep — PASS
- `python src/data_prep.py --split-seed 42` -> 1034 tunes, dropped_no_key=0,
  train 930 / val 104 tunes (90/10, seed 42), train_win=287862,
  val_win=34370 (seq_len 64, stride 1).
- Unit checks: vocab round-trip OK; (N,64)+scalar shapes OK;
  windows never cross tune boundary (built per tune); no tune in both
  splits (asserted); stateful divisibility asserted in train.py.

## 2026-10-08 — Phase 3: baselines — PASS
- `python src/evaluate.py --baselines` -> most-frequent acc 0.155 /
  ppl 2091.1 (0.99-on-top rule, stated); char 4-gram + add-1 smoothing
  acc 0.521 / ppl 6.1.
- `python src/train.py --config exp --model small_rnn --tag
  baseline_small_rnn` -> val acc 0.543 / ppl 4.6 (41,430 params).
  (Fixed `build()` kw-filter bug for small_rnn; logged, no data touched.)

## 2026-10-08 — Phase 4: main Char-RNN — PASS
- Smoke (`--config smoke`, 15%/5ep, GPU): val acc 0.585, ~19 s/epoch.
  Already beats all baselines.
- Full (`--config full --tag main_full`, 100%/stride-2/batch-128/max-60ep,
  patience 8): early stop ep 19 (best val ep 11) -> train 0.635/0.783/1.89,
  val 1.049/0.675/2.85. Beats baselines. Checkpoint + history saved.
- Curves `outputs/figures/curves_main_full.png` plotted from saved history.

## 2026-10-08 — Phases 5-6: variants — PASS
- `var_many2many` val 2.848/0.587/17.3 (loss not comparable: all positions);
  `var_stateful` 1.438/0.549/4.2 (shuffle=False, per-epoch reset in log);
  `var_cnn` 1.727/0.510/5.6. One sample tune each (Phase 7).
  (Fixed `build()` batchnorm kw for many2many/cnn; E4 is main-model-only.)

## 2026-10-08 — Phase 7: generation + conversion — PASS
- INSTRUMENT BUG + FIX: first 24-attempt run scored 0/24 because (a) a
  `data_prep` join artifact left a leading `\n` on 929/930 tunes (validator
  `startswith X:` failed; repair double-headered), and (b) the repeat rule
  (strict equality) was wrong for the corpus. Measured corpus (930 tunes):
  closers >= openers always (584 more_close, 346 equal, 0 more_open);
  endings are `:|`/`|`, never `|]`. Rule corrected to openers <= closers
  (PRD 11.1 allows stating the corpus convention), seeds stripped, stop
  tokens `:|`/`|]`. Buggy-validator attempts discarded (instrument, not
  model result); block reset and re-run clean.
- `python src/generate.py --checkpoint .../main_full.pt --temperatures 0.6
  0.8 1.0 1.2 --attempts-per-temp 6` -> 24 attempts, 24 valid (19 raw +
  5 repaired), rate 1.00, 6/6 per temperature. Deterministic (torch seed).
- `python src/convert.py` -> 23/23 main + 3 variant + tabla = 28 triplets
  (.abc/.mid/.wav), WAVs asserted non-silent. 28/28 original (exact-match).
- Tabla demo `experiments/tabla_demo.py` -> `gen_tabla_01.*` (status raw):
  extension demo only, stated in report.

## 2026-10-08 — Phase 8: experiments + report — PASS
- `python experiments/run_all.py`: 26 GPU runs done (E1x7 E2x2 E3x3 E4 E6x2
  E7x2 E11x5 E12_gpu + 3 variants). E12: CPU 54.4 vs GPU 3.9 s/epoch
  (~13.9x). E5 recorded as analysis (labeled); E8/E9/E10 as reuse (labeled).
- `python src/evaluate.py --final` -> note_compare.png, experiments.csv
  (37 rows), baselines synced. `schema_check.py` -> results.json PASS.
- Report `report/report.md` complete (Ch.1-9 incl. viva sheet, failure
  gallery `report/failure_gallery.*` from smoke checkpoint, listening
  table with computed structure + blank session column).
- README clean-run test (isolated copy, smoke scale): data_prep PASS,
  train smoke PASS (val 0.579), generate 6 attempts 6/6 valid, convert 6/6,
  evaluate --all PASS. Substitutions logged: smoke config instead of full
  (13-min full already exercised in main build), 2 temps; venv/requirements
  install skipped (pinned env reused) — commands themselves verbatim.

