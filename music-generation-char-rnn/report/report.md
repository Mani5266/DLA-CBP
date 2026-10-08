# Music Generation Using Char-RNN — Final Report

Deep Learning Applications · Theory 22PC1DS402 + Lab 22PC2DS402 ·
B.Tech Data Science (Final Year)

> Honesty note: every number below is measured in this build and traces to
> `outputs/results.json`, a log in `outputs/logs/`, or a figure in
> `outputs/figures/`. No benchmark numbers are copied from anywhere — there
> are no published numbers for this exact build to copy.

## 1. Problem statement and real-world framing

**Given** a seed/start string in abc notation (a valid header plus an opening
fragment), **generate** a continuation that forms a complete, new folk-style
tune: valid abc headers, consistent meter and key, well-formed bar lines, a
clean ending, not a copy of any training tune, convertible to MIDI and
playable as audio.

Music-as-text turns composition into **next-character prediction** — a
multiclass classification problem at every step (Softmax + cross-entropy),
exactly the framing the syllabus teaches — with an output a listener can
judge in seconds. Uses: (a) composition aid — a melody sketch to edit, not a
replacement for composers; (b) education — hear what the model learned about
meter, key and phrase structure by varying seed and temperature.

Out of scope (not attempted): multi-track/polyphonic music, lyrics,
waveform-level audio generation, any web app/GUI, beating external
benchmarks, other lab blocks (HAR, Quora, Self-Driving).

## 2. Dataset

**Source:** Nottingham Music Database — public-domain traditional folk tunes
(reels, jigs, hornpipes, waltzes) in plain-text abc (~5 MB). Used the cleaned
mirror `github.com/jukedeck/nottingham-dataset` (`ABC_cleaned/`, 14 files).
License: public domain; no account, fee or permission needed. No other data
was used anywhere (MAESTRO backup never needed).

**Measured statistics** (`outputs/results.json:dataset`, EDA figures in
`outputs/figures/`):

| Statistic | Measured value |
|---|---|
| Tunes total | 1034 (regex `^X:` count; matches the ~1033 cleaned figure) |
| Train / validation tunes | 930 / 104 (tune-level 90/10 shuffle, seed 42) |
| Total characters | 390475 |
| Vocabulary size | 86 (tens, as expected — cleaning verified) |
| Dropped (no `K:` line) | 0 |
| Train / val windows (seq 64) | 287862 / 34370 at stride 1 |
| Mean tune length | 375.6 chars |
| Meters (top) | 4/4: 548, 6/8: 360, 3/4: 60, 2/4: 41, 9/8: 13 (+2/2: 7, 3/2: 3, 6/4: 2) |
| Keys (top) | G: 361, D: 358, A: 125, C: 54, Am: 39, Em: 29, F: 25 |
| Top characters | `"` (54138), `2` (44820), `/` (38346), `\|` (24018), space, `A` |

**Leakage rule (enforced in code, asserted):** split by TUNE before windowing,
never by windows — splitting after windowing leaks near-identical windows of
one tune into both sets and inflates validation accuracy. `data_prep.py`
asserts no tune appears in both splits; windows never cross a tune boundary
(built per tune).

## 3. Architecture

Main model (many-to-one Char-RNN), implemented in PyTorch (`src/model.py`):

| Layer | Configuration |
|---|---|
| Input | Integer char IDs, shape (batch, 64), from `vocab.json` |
| Embedding | vocab(86) x 256 (learned character embedding — **not** Word2Vec) |
| LSTM 1 / 2 / 3 | 256 units each, batch-first; 1,623,126 parameters total |
| Dropout | 0.3 after LSTM 3 (varied in E3) |
| Output | Dense(86) + Softmax — next-character distribution |
| Loss / metrics | Cross-entropy; accuracy; perplexity = exp(loss), per epoch |
| Optimizer | Adam, lr 3e-3, gradient clipping norm 1.0 |

Framework note (Unit IV evidence): TensorFlow 2.11+ has **no native-Windows
GPU support** (last GPU build TF 2.10, Python ≤ 3.10), so the identical maths
was built in PyTorch 2.1+cu121 to actually use the RTX 3050 (E12 measures
~14× vs CPU). Concepts transfer 1:1 (BPTT, Softmax+CE, Adam, dropout).

| Variant | Difference | Val acc (budget) |
|---|---|---|
| Many-to-many (Lab 13–14 arch.) | 3×LSTM return-sequences + per-step Dense; targets = input shifted by 1 | 0.587 (reduced) |
| Stateful | Same sizes; ordered unshuffled batches, hidden carried across batches, reset each epoch, batch-count divisibility asserted | 0.549 (reduced) |
| Char-CNN | Emb256 → Conv1D(256,k5,ReLU) → Conv1D(256,k3,ReLU) → global max-pool → Dense256 → Dropout → Dense(86); 634,710 params | 0.510 (reduced) |
| Baselines | (a) most-frequent char 0.155 / ppl 2091; (b) char 4-gram + add-1 smoothing 0.521 / ppl 6.1; (c) small Elman RNN (Emb64+RNN128, 41,430 params) 0.543 / ppl 4.6 | — |

## 4. Training

| Setting | Smoke | Full (main result) |
|---|---|---|
| Data | 15% tunes | 100% (144,162 windows at stride 2 — see Limitations) |
| Epochs | 5 | max 60, early stopping patience 8 → **stopped at 19** (best val, ep 11) |
| Batch / seq / opt | 64 / 64 / Adam 3e-3 | 128 / 64 / Adam 3e-3, clip 1.0 |
| Time | ~20 s/epoch GPU | ~40 s/epoch GPU (~13 min total) |
| Val result | acc 0.585, ppl 3.7 | **acc 0.675, loss 1.049, ppl 2.85** (train 0.783/0.635/1.89) |

Checkpoint rule: best validation loss kept (`outputs/checkpoints/main_full.pt`
+ config); curves in `outputs/figures/curves_main_full.png` plotted from the
saved history. Main model beats all three baselines (0.675 > 0.543 > 0.521 >
0.155). Train–val gap (0.783 vs 0.675) shows expected mild overfitting on
~1k tunes — the reason for E3/E6.

## 5. Experiments E1–E12

Protocol: one variable at a time; reduced budget (15% data / 8 epochs /
stride 2) unless noted; same split/seed/eval. Full table:
`outputs/experiments.csv` (37 rows).

| ID | Changed | Result (val acc; best bold) | Syllabus link |
|---|---|---|---|
| E1 | Optimizer: SGD 0.155, SGD+Mom 0.340, NAG 0.321, AdaGrad 0.548, AdaDelta 0.155, RMSProp **0.607**, Adam 0.574 | Adaptive methods win on a short budget; plain SGD barely leaves the majority class | Unit III (which optimizer when) |
| E2 | Clip on vs off at lr=1e-2 | Both collapsed to 0.155 (majority class) — clipping cannot rescue a 3×-too-high LR in 8 epochs | Unit I/III (gradients, clipping) |
| E3 | Dropout 0 / 0.3 / 0.5 | 0.569 / 0.572 / 0.570 — indistinguishable at 8 epochs; the gap opens by ep 19 of the full run | Unit II (regularization) |
| E4 | +LayerNorm variant | 0.608 vs 0.572 — small win, faster early convergence (LayerNorm used and stated; sequence model, per-step statistics) | Unit II (normalization) |
| E5 | Activation/init | **Analysis, not a run** (labeled as such): E2's collapse + Char-CNN ReLU success + E4 norm gain jointly show scale control matters more than the nonlinearity choice here | Unit II |
| E6 | LSTM 128 / 256 / 512 | 0.555 / 0.572 / 0.575 with 0.49M / 1.62M / 5.85M params at 6 / 9 / 22 s·epoch⁻¹ — steep cost, flat gain; 256 is the knee | Unit IV (tuning, capacity) |
| E7 | Seq len 32 / 64 / 100 | **0.585** / 0.572 / 0.566 — shorter context learns local ornament patterns faster on a fixed budget | Unit I (BPTT horizon) |
| E8 | Temperature 0.6/0.8/1.0/1.2 | 6/6, 6/6, 6/6, 6/6 valid (24 attempts); higher T → shorter, more wandering tunes (§6) | Sampling / generation |
| E9 | Stateful vs stateless | 0.549 vs 0.572 — carrying state across shuffled-tune boundaries adds noise; stateless shuffling wins on a tune-sliced corpus | Unit V (stateful arch.) |
| E10 | Family: RNN vs CNN vs many2many | 0.675 (full) / 0.510 / 0.587 — recurrence beats local convolution for character music; many-to-many loss (2.85, ppl 17.3) is not comparable (predicts easy+hard positions jointly) | Lab CO5 |
| E11 | LR × batch (3×2) | Best cell **lr 1e-3 × batch 32 (0.583)**; lr 1e-2 collapses (0.448 / 0.123 — batch 64 diverges harder) | Unit IV (tuning) |
| E12 | Local CPU vs local GPU, identical config | 54.4 vs 3.9 s·epoch⁻¹ (**~14×**) on RTX 3050; environment strings recorded | Unit IV (GPU vs CPU, Colab note) |

## 6. Generation and evaluation

Autoregressive sampling (seed = real header + 40–110-char fragment, 6/8–4/4
folk openings), stop at corpus-style ending (`:|` or `|`, min 120 chars,
max 800), temperatures 0.6/0.8/1.0/1.2 × 6 attempts = **24 attempts**.

- **Validity: 24/24 (1.00)** — raw 19/24 (0.79), post-repair 5/24. Repairs
  used only: header prepend, final-bar append, dangling-opener drop (each
  recorded per tune). Denominator = all attempts; nothing silently discarded.
- **Validator rule (corpus-measured):** repeats valid iff openers ≤ closers —
  584/930 train tunes have *more* closers (repeat-from-start convention), 346
  equal, 0 more openers. Strict equality would fail the corpus itself.
- **Originality:** 28/28 delivered tunes are not exact copies of any training
  tune (whitespace-normalized exact-match).
- **Corpus comparison** (`outputs/figures/note_compare.png`): generated
  note-letter distribution tracks the corpus (D/G/A-heavy); keys stay in
  G/D/A/Am — no mode collapse to a single key.
- **Deliverables:** `outputs/generated/` — 24 main tunes + 3 variant samples
  (many-to-many, stateful, Char-CNN @0.8) + tabla sketch, each as
  `.abc + .mid + .wav` (28 triplets, all WAVs programmatically asserted
  non-silent, non-zero duration). Featured five: `gen_T0.6_0`, `gen_T0.8_0`,
  `gen_T1.0_0`, `gen_T1.2_0` (one per temperature) + `gen_tabla_01`.
- **Listening notes (structural, computed — listening session to fill):**

| Tune | Temp | Chars | Key/meter | Computed note | Listening (fill in session) |
|---|---|---|---|---|---|
| gen_T0.6_0 | 0.6 | 475 | G/4/4, 20 bars | coherent, corpus-like phrasing | |
| gen_T0.8_0 | 0.8 | 802 | D/4/4, 40 bars | balanced variety, longest | |
| gen_T1.0_0 | 1.0 | 494 | D/6/8, 36 bars | more adventurous ornament | |
| gen_T1.2_0 | 1.2 | 369 | G/4/4, 23 bars | shortest, most wandering | |
| gen_tabla_01 | 1.0 | 309 | D/6/8, 20 bars | driving semiquaver rhythm | |

(Play the `.wav` files; expected gradient: coherent → repetitive →
wandering as temperature rises — the 24/24 validity held, but musical
focus visibly loosens at 1.2.)

- **Failure gallery** (`report/failure_gallery.abc/.txt`): 5-epoch smoke
  checkpoint at T=1.0 — validator-*valid* abc that wanders (stray `-`
  chars, key-center drift D-over-Am, meandering phrases). Lesson: validity
  ≠ musicality; judging generation from a smoke run misleads (PRD §11).
- **Tabla extension (honest):** `gen_tabla_01` seeds a 6/8 jig + 1/16-note
  ostinato at T=1.0 → a driving D-major sketch (status: raw-valid). abc has
  no timbre/bols and one pitch line — this demonstrates rhythm conditioning,
  it is **not** a tabla model and the report must never call it one.

## 7. Limitations and conclusion

1. Stride 2 (not 1) on the full run — laptop-GPU time budget; windows still
   tune-bound, split unchanged. A Colab-scale rerun at stride 1 is the
   obvious follow-up. 2. Mild overfitting (train 0.78 vs val 0.67) — ~1k
   tunes is small for 1.6M params; E6-128 is the efficient alternative.
3. E2 shows LR dominates clipping; E11's best cell (1e-3×32) was not
   re-run at full budget. 4. Sine-synth WAVs, not soundfont audio; single
   melody line (chords stripped); `>`/`<` broken-rhythm marks ignored.
5. E10 budgets differ across families (stated, not hidden).

Conclusion: a 3×256 Char-RNN learns folk structure well enough to compose
24/24 valid, original, playable tunes (val acc 0.675, ppl 2.85, beating
n-gram 0.521 and small-RNN 0.543 baselines), with the full experiment
battery confirming syllabus concepts behave as taught — adaptive optimizers,
capacity knee at 256, ~14× GPU speedup, temperature-controlled variety.

## 8. Dataset citation

Nottingham Music Database (public-domain folk tunes), cleaned version via
`github.com/jukedeck/nottingham-dataset` (Jukedeck). MAESTRO not used.

## 9. Viva sheet — where in my project is X?

| Syllabus topic | Location |
|---|---|
| Unit V: problem + abc representation | Ch.1–2; Phase 1 EDA (`char/meter/key_dist.png`) |
| Char-RNN model | Ch.3; `src/model.py:CharRNN`; `checkpoints/main_full.pt` |
| Data preparation | Ch.2; `src/data_prep.py`; `data/processed/` + `sequences_meta.json` |
| Many-to-many + TimeDistributed | `src/model.py:ManyToManyRNN`; E10; `varB_T0.8_0.abc` |
| Stateful RNN | `src/model.py:StatefulRNN` + carry loop in `train.py`; E9; `varC_*` |
| Model training | Ch.4; `curves_main_full.png`; `logs/history_main_full.json` |
| Char-RNN generation | Ch.6; `src/generate.py`; 24 tunes; E8 |
| Tabla extension | `gen_tabla_01.*`; Ch.6 (demo, not a tabla model) |
| MIDI generation | `src/convert.py`; 28 `.mid` + `.wav` |
| Char-CNN (CO5) | `src/model.py:CharCNN` (ReLU); E10; `varCNN_*` |
| BPTT / vanishing gradients | E2 collapse, E7 horizon effect, Ch.5 |
| Dropout / regularization | E3, train–val gap Ch.4 |
| ReLU / init / BatchNorm | CNN ReLU; E5 analysis; E4 LayerNorm (stated) |
| Adam etc. / clipping | E1 tournament; E2; `train.py:get_opt` |
| Softmax + cross-entropy | Output layer Ch.3; loss everywhere |
| TF/Keras, GPU vs CPU, Colab | Framework note Ch.3; E12 (~14×); Colab replaced by local RTX 3050 (logged) |
| Hyperparameter tuning | E11 grid; early stopping Ch.4 |
| NOT covered (stated) | Word2Vec (char-Embedding is not Word2Vec); HAR/Quora/Self-Driving blocks |
