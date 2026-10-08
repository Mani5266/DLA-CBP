Music Generation Using Char-RNN - PRD

Product Requirements Document - Agent-Facing Build Guide

# Music Generation Using Char-RNN

One complete build document. An AI coding agent - OpenCode, Muse, or equivalent - executes this PRD phase by phase, end to end, with no further questions, and produces a trained model, playable generated tunes, measured results, and a submission-ready report.

Theory 22PC1DS402 · Lab 22PC2DS402 · Deep Learning Applications · B.Tech Data Science - Final Year

| Item | Specification in this PRD |
|---|---|
| **Exact project title** | Music Generation Using Char-RNN |
| **Input** | A seed / start string in abc notation |
| **Output** | A new, original folk-style tune: valid .abc, converted .mid and playable .wav |
| **Dataset** | Nottingham Music Database - public domain, plain-text abc, about 5 MB. No paid or restricted data anywhere in this build. |
| **Main model** | Embedding (vocab x 256) - LSTM 256 x 3 - Dropout - Dense (vocab) with Softmax. Sequence length 64, next-character prediction. |
| **Also built** | Many-to-many TimeDistributed variant, stateful RNN variant, Char-CNN comparison, n-gram baseline, 12 controlled experiments (E1-E12). |
| **Student hardware** | Windows laptop, NVIDIA RTX 3050, Python 3.11. GPU training runs on Google Colab - see the platform trap in Section 5. Do not plan native-Windows GPU training. |
| **Definition of success** | Section 12 checklist fully met, with every number in the final report traceable to an actual run recorded in results.json. |

![End-to-end pipeline diagram from dataset download through evaluation](.src/media/pipeline.png)

**Read this first - the honesty rule.** This PRD deliberately contains **no pre-filled accuracy, loss, or perplexity numbers**. There are no published benchmark numbers for this exact build to copy, and inventing them is fabrication. Expected outcomes are stated qualitatively only (Section 9). Every result table is filled from real runs.

## 1 Operating rules for the building agent

These rules override convenience, speed, and the agent's own defaults. If a rule and a shortcut conflict, the rule wins.

### 1.1 Work phase by phase, gate by gate

1. Execute Phases 0-8 in order (Section 7). Never start a phase whose predecessor's exit criteria have not passed.
2. At the end of each phase, run every exit-criteria check, print the result, and record pass/fail in outputs/phase_log.md with the date, command run, and output summary.
3. If a check fails, stop. Do not paper over it, skip it, or lower the threshold silently. Diagnose the root cause, fix it, and re-run the whole phase check.

### 1.2 Never fabricate results or metrics

- Every number in the report, README, charts, and results.json must come from a file, log, or metric produced by an actual run in this build.
- Do not copy numbers from blogs, papers, tutorials, or this PRD. This PRD contains no target numbers to copy on purpose.
- If a run fails, crashes, or is not executed, record it as "status": "failed" or "not_run" with the reason. A failed experiment honestly reported scores better than a fake success.
- Plots must be generated from the saved training history / metrics files, never hand-drawn to look plausible.

### 1.3 Save everything to the output contract

- All deliverables go to the exact tree in Section 10. No result lives only in a notebook cell, terminal scrollback, or chat message.
- After each training or experiment run, immediately append/update its entry in outputs/results.json (schema in Section 10.3) before starting anything else.
- Keep raw logs: one log file per run under outputs/logs/, named <phase_or_exp>_<config>.log.

### 1.4 Failure protocol

1. Log the exact error, the command, and the environment (local / Colab) in outputs/phase_log.md.
2. Fix the root cause - wrong path, wrong shape, wrong version - not the symptom. Do not delete failing tunes, sequences, or test cases to make a check pass.
3. Retry the same step once after the fix. If it still fails, try the documented fallback in this PRD (for example, the mirror download in Section 4.2, or music21 instead of abc2midi in Section 5). Only if all fallbacks fail, mark the step blocked, state exactly what is blocked and why, and continue with phases that do not depend on it.

### 1.5 Data and cost rules

- **No paid, gated, or restricted datasets.** Nottingham only; MAESTRO (MIDI-only) only as the documented backup. No Kaggle-gated downloads, no request forms.
- **No paid services.** Free Colab tier is sufficient by design: experiments run on reduced budgets (Section 8), only the main model gets the full run.
- Pin versions in requirements.txt. Do not upgrade TensorFlow mid-build to chase a feature.
- Set and record a fixed random seed (use 42) for data splitting and training, and record it in results.json.

**Cheap-first principle.** Always run the FAST smoke config (Section 6.3) before any full run. A 5-epoch smoke run that proves download -> prep -> train -> generate -> convert works is worth more than a 100-epoch run that fails at conversion.

## 2 Mission and problem statement

### 2.1 Mission

Build a character-level recurrent neural network that learns the structure of traditional folk melodies written in abc notation, and then composes new, original folk-style tunes - character by character - that are syntactically valid abc, convertible to MIDI, and playable as audio.

### 2.2 Problem statement (formal)

**Given** a seed / start string in abc notation (for example a valid header plus an opening fragment), **generate** a continuation that forms a complete, new tune: it has valid abc headers, consistent meter and key, well-formed bar lines, and a clean ending; it is **not** a copy of any training tune; and it can be converted to MIDI and rendered to playable audio without errors.

### 2.3 Why this is a real problem

- **Composition aid:** a working draft generator gives composers and arrangers starting material in a folk style - a melody sketch they can edit, not a finished product that replaces them.
- **Education:** students of folk music and of deep learning can hear what a model has actually learned about meter, key, and phrase structure, and can probe it live by changing the seed and the sampling temperature.
- **A clean deep-learning problem:** music-as-text turns composition into next-character prediction - a multiclass classification problem at every step (Softmax + cross-entropy) - which is exactly the framing the syllabus teaches, with an output a listener can judge in seconds.

### 2.4 Scope and non-goals

| In scope | Out of scope |
|---|---|
| Single-melody folk tunes in abc notation; Char-RNN (main), many-to-many and stateful variants, Char-CNN comparison; temperature-controlled generation; abc -> MIDI -> WAV; quantitative + listening evaluation; experiments E1-E12. | Multi-track / polyphonic composition; lyrics generation; audio-waveform generation (no WaveNet-style models); a web app or GUI product; beating any external benchmark; any other lab block (HAR, Quora, Self-Driving). |

## 3 Syllabus mapping - what this build covers

Subjects: Deep Learning Applications theory **22PC1DS402** and Deep Learning Applications Laboratory **22PC2DS402**. The table below is the contract: every row names where in the build the topic is demonstrated, so the final report's viva sheet (Section 10) can point to a file, figure, or experiment for each one.

| Syllabus item | Coverage | Where it is demonstrated in this build |
|---|---|---|
| **Theory Unit V** - Music Generation case study | **FULL** | The entire project. See the Unit V breakdown table on the next page - every named sub-topic has a phase. |
| **Lab Weeks 13-14** - Music Generation | **FULL** | Phases 1-7 cover items (a)-(g) of Weeks 13-14 one-to-one (mapping on next page). |
| **Lab CO4** - music generation concepts | **FULL** | Sections 2, 4, 6, 7 (Phases 4-7) and the generated-tune deliverables. |
| **Lab CO5** - design models with RNN & CNN | **FULL** | Char-RNN family (Phase 4-5) + Char-CNN comparison (Phase 6), compared head-to-head in E10. |

### 3.1 Unit V and Lab Weeks 13-14, sub-topic by sub-topic

| Named syllabus sub-topic | Build location |
|---|---|
| Real-world problem and music representation | Section 2; Phase 1 EDA (abc as a text representation of music) |
| Char-RNN with abc notation - model | Section 6; Phase 4 (main model) |
| Char-RNN with abc notation - data preparation | Section 4.5-4.6; Phase 2 |
| Many-to-many RNN, TimeDistributed Dense layer | Section 6.2; Phase 5, variant B; compared in E10 |
| Stateful RNN, model architecture | Section 6.2; Phase 5, variant C; compared in E9 |
| Model training | Phase 4 (full run) + training curves deliverable |
| Char-RNN music generation | Phase 7; temperature study E8 |
| Tabla music generation (extension) | Phase 7, step 7.6: generation conditioned on a rhythm-style seed / extension note in the report. This is an extension demonstration, not a separately trained tabla dataset model - the report must say so honestly. |
| MIDI music generation | Phase 7 conversion (abc -> MIDI -> WAV); .mid deliverables |
| Char-CNN model (Lab Week 13-14, item b) | Phase 6; compared in E10 |

### 3.2 Units I-IV: how the earlier theory is used

| Unit | Topics used | Demonstration |
|---|---|---|
| **Unit I** | Perceptron / MLP basics (framing); backpropagation - here, backpropagation through time (BPTT); vanishing / exploding gradient problem | Framing in report Ch. 2; BPTT explained for the LSTM; gradient behaviour shown live in E2 (clipping on/off at high learning rate) and in training-curve discussion |
| **Unit II** | Dropout and regularization; ReLU (framing / CNN); weight initialization; Batch Normalization (variant); SGD, SGD + momentum, Nesterov (NAG) | E3 (Dropout), E4 (BatchNorm variant), E1 (SGD family), E5 (init / activation analysis); ReLU is the Char-CNN activation in Phase 6 |
| **Unit III** | AdaGrad, AdaDelta, RMSProp, Adam - which optimizer when; gradient checking and clipping; Softmax + cross-entropy for multiclass classification | E1 (full optimizer tournament), E2 (clipping); the main model's output layer *is* Softmax + categorical cross-entropy, one class per vocabulary character |
| **Unit IV** | TensorFlow and Keras; GPU vs CPU; Google Colaboratory; hyperparameter tuning in Keras | Whole build is TensorFlow/Keras; E12 (local CPU vs Colab GPU timing); E11 (LR x batch-size grid); Colab setup in Section 5.3 |

**Honest exclusions - state these in the report, do not hide them.** This build does **not** cover Word2Vec (CBOW / Skip-gram, Unit III) - there is no word-embedding task in a character-level music model, and the learned character Embedding layer is not Word2Vec; do not claim it is. It also does not cover the other lab blocks: Human Activity Recognition (Weeks 5-8), Quora Question Pairs (Weeks 9-12), or Self-Driving Cars (Weeks 1-4). One project cannot cover every block well; this one covers Unit V and Lab Weeks 13-14 fully instead.

## 4 Dataset bible - Nottingham Music Database

| Fact | Detail |
|---|---|
| **Name** | Nottingham Music Database |
| **Content** | Traditional folk tunes (reels, jigs, hornpipes, waltzes and others) in abc notation - plain text. About 1,200 tunes in the original collection; the corrected / cleaned GitHub version contains about 1,033 tunes. The exact count found at download time is recorded in Phase 1 EDA - report the measured count, not either figure above. |
| **License** | The folk tunes are public domain. No permission, account, or fee is required. |
| **Size** | Tiny - about 5 MB. Downloads and loads in seconds; the whole corpus fits in memory. |
| **Primary source** | ifdo.ca/~seymour/nottingham (also listed at abc.sourceforge.net/NMD) |
| **Cleaned mirror (preferred for the build)** | github.com/jukedeck/nottingham-dataset - cleaned abc files with consistent chord, repeat, and part notation, plus MIDI conversions. Use this if the primary source layout differs from Section 4.3. |

### 4.1 Download steps (building agent)

1. Try the cleaned mirror first: git clone https://github.com/jukedeck/nottingham-dataset into data/raw/. If git is unavailable, download the repository ZIP from the same page and extract it there.
2. Fallback: download the abc collection from the primary source above and place the .abc files in data/raw/.
3. Verify: count .abc files, open three at random, confirm each starts with an X: header and contains a K: line and bar symbols (|). Record the count and file list in outputs/phase_log.md.
4. Never re-upload or redistribute the dataset as your own work; cite it in the report (Section 10.4) as public-domain folk tunes via the Nottingham Music Database / Jukedeck cleaned version.

### 4.2 Expected layout after download

```
data/
  raw/                  # as downloaded, NEVER edited in place
    *.abc               # one file may contain one or more tunes
  processed/
    corpus.txt          # cleaned, concatenated corpus (Phase 2)
    train.txt  val.txt  # tune-level split files (Phase 2)
    vocab.json          # character vocabulary + index maps
    sequences_meta.json # seq_len, counts, seed, split sizes
```

### 4.3 Backup dataset - only if Nottingham fails completely

**MAESTRO - MIDI-only download only, about 57 MB, CC BY-NC-SA 4.0.** Use it only if both Nottingham sources are unreachable after the fallback steps, and record the switch and the reason in the phase log and the report. **Never download the MAESTRO audio version (about 87-122 GB)** - it will fill the disk and is not needed. If MAESTRO is used, the build changes representation from abc characters to MIDI events; that is a different project shape, so flag it as a scope change wherever results are reported.

### 4.4 Anatomy of abc notation

abc is a text format: header fields (one letter, a colon, a value) followed by the tune body - note letters, durations, bar lines, and repeat signs. A short public-domain folk-tune header looks like this in structure:

```
**X:**1            tune / reference number
**T:**Example Reel  title
**M:**4/4           meter (time signature)
**L:**1/8           default note length
**R:**reel          rhythm / type (optional)
**K:**G            key - header ends, tune body follows
GABc d2 Bd | e2 ge dBGB | ... |]
```

| Field | Meaning | Why the model / validator cares |
|---|---|---|
| X: | Reference number | Every generated tune must start with one - first validity check |
| T: | Title | Required header; generated tunes get generated / placeholder titles |
| M: | Meter, e.g. 4/4, 6/8 | Meter distribution is an evaluation statistic (Section 9) |
| L: | Default note length | Controls how bare note letters are read as durations |
| K: | Key, e.g. G, D, Amin | Last header field; key distribution is an evaluation statistic |
| \| \|] \|: :\| | Bar line, final bar, repeat start/end | Bar structure and clean termination are validity checks; unmatched repeats are the top conversion failure (Section 11) |

### 4.5 Cleaning rules (Phase 2 - apply exactly)

- Work on a copy in data/processed/; never edit data/raw/.
- Split the raw files into individual tunes on the X: line at the start of a line. A tune with no K: line is corrupt - drop it and count it.
- Remove comment lines (starting with %) and non-musical metadata lines as follows: drop W: (lyrics / words) and Z: (transcriber) lines. Keep X: T: M: L: R: K: headers and the full tune body.
- **Keep musical symbols:** note letters A-G / a-g, accidentals ^ _ =, duration digits, | : [ ], chord symbols in quotes, spaces, and newlines. Do not lowercase the corpus - in abc, case is pitch (uppercase and lowercase are different octaves).
- Normalize line endings to \n, collapse 3+ blank lines to one blank line (tune separator), and strip trailing spaces.

### 4.6 Vocabulary, split, and sequences - the leakage rules

- **Vocabulary:** the sorted set of unique characters in the cleaned corpus. Build char2idx / idx2char, save to vocab.json. Expect a small vocabulary (tens of characters, not thousands) - record the measured size; if it is in the thousands, cleaning has failed.
- **Split by TUNE, never by character windows:** shuffle tunes with seed 42, split about 90% train / 10% validation *as whole tunes*, then build sequences inside each split. Splitting after windowing leaks near-identical windows of the same tune into both sets and inflates validation accuracy - this is the dataset equivalent of the HAR leakage trap and it invalidates Section 9.
- **Many-to-one sequences (main model):** input = 64 consecutive characters, target = the single next character (integer class). Stride 1 inside a tune; never let a window cross a tune boundary.
- **Many-to-many sequences (variant):** input = 64 characters, target = the same window shifted by one character at every position, fed to TimeDistributed Dense. Same tune-boundary rule.
- **Stateful variant ordering:** sequences must be in corpus order, grouped so that each batch row continues the previous batch's text; use shuffle=False, a batch size that divides the sequence count, and reset states at each epoch end. Any shuffle breaks statefulness silently - assert it in code.

## 5 Tech stack and environment setup

| Layer | Choice (pinned in requirements.txt) |
|---|---|
| Language | Python 3.11 |
| Deep learning | TensorFlow / Keras (single version pinned for the whole build; record the version in results.json) |
| Numerics / data | numpy; standard library for text processing (no heavy NLP stack needed for character data) |
| Plots | matplotlib |
| abc -> MIDI | Primary: abc2midi (abcMIDI tools). Fallback: music21 abc parsing to MIDI. Use whichever installs cleanly; record which one produced the deliverables. |
| MIDI -> WAV | TiMidity++ or FluidSynth with a General MIDI soundfont. On Colab, install via apt in the setup cell; locally, any one of the two is enough. |
| GPU training | Google Colab (free tier) - see 5.3 |
| Local training | CPU only (smoke runs, data prep, generation, conversion, evaluation) |

**Platform trap - TensorFlow on Windows.** TensorFlow 2.11 and later have **no native-Windows GPU support**; the last version with it was 2.10. The student's RTX 3050 therefore does **not** accelerate TensorFlow in native Windows Python 3.11. Do not waste build time trying to force it. The supported routes are: (a) **Colab GPU for training** - the default in this PRD; (b) local CPU for everything else; (c) WSL2 with Linux TensorFlow, only if the student already has WSL2 set up - optional, never a blocker. Experiment E12 is defined accordingly: **local CPU vs Colab GPU**, same config, and the report states this reason explicitly - that explanation is itself Unit IV evidence.

### 5.1 Repository tree (exact - create it in Phase 0)

```
music-generation-char-rnn/
  data/raw/  data/processed/
  src/
    data_prep.py    # cleaning, vocab, splits, sequences
    model.py        # main / many-to-many / stateful / Char-CNN builders
    train.py        # training entry point (config-driven)
    generate.py     # temperature sampling + abc validation/repair
    evaluate.py     # metrics, validity rate, corpus statistics
    convert.py      # abc -> MIDI -> WAV
  experiments/      # one config / script per experiment E1-E12
  outputs/
    results.json  phase_log.md  logs/
    figures/        # EDA + training curves + experiment plots
    generated/      # .abc + .mid + .wav tunes
    checkpoints/    # best model weights
  report/           # final project report + viva sheet sources
  requirements.txt  README.md
```

### 5.2 Local setup steps

1. Create a virtual environment (python -m venv .venv), activate it, install requirements.txt.
2. Verify: print TensorFlow version and confirm it imports; print whether a GPU is visible - on native Windows the expected, correct answer is *none* (see trap above). Record both in the phase log.
3. Verify conversion tools: run the chosen abc -> MIDI tool on one dataset tune and the MIDI -> WAV tool on the result; keep the test output as Phase 0 evidence.

### 5.3 Colab setup steps (for every GPU session)

1. Runtime -> Change runtime type -> GPU. Verify with nvidia-smi and TensorFlow's GPU list; log the GPU name.
2. Mount / upload the project folder, install requirements.txt (skip anything already provided by Colab, but record substitutions), install conversion tools if generation checks run there.
3. Run training from src/train.py with a config - never as notebook-only cells - and download outputs/ (checkpoints, logs, results.json updates) at the end of the session. Colab sessions are disposable; anything not downloaded is lost.

## 6 Model architecture

![Main Char-RNN architecture diagram](.src/media/architecture.png)

### 6.1 Main model - many-to-one Char-RNN (exact)

| Layer | Configuration | Notes |
|---|---|---|
| Input | Integer character IDs, shape (batch, 64) | From vocab.json; no one-hot input needed with Embedding |
| Embedding | vocab_size x 256 | Learned character embedding - not Word2Vec (Section 3.2) |
| LSTM 1 | 256 units, return_sequences=True | Feeds LSTM 2 a full sequence |
| LSTM 2 | 256 units, return_sequences=True | Feeds LSTM 3 a full sequence |
| LSTM 3 | 256 units, return_sequences=False | Last-step state summarises the 64-char context |
| Dropout | 0.3 in the main config (varied in E3) | Applied after LSTM 3, before Dense |
| Dense + Softmax | vocab_size units, Softmax | Probability distribution over the next character |
| Loss | Categorical cross-entropy (sparse form for integer targets is acceptable - state which was used) | Next-character multiclass classification |
| Metrics | Accuracy; perplexity = exp(loss), computed and logged per epoch | Both recorded for train and validation |
| Optimizer (main) | Adam, default Keras learning rate unless E11 finds better | Gradient clipping value set in config (see E2) |

### 6.2 Required variants (Phase 5-6 - same data, same split)

| Variant | Difference from main model |
|---|---|
| **Many-to-many** (Phase 5B) | LSTM stack with return_sequences=True throughout, then TimeDistributed(Dense(vocab, Softmax)) - a prediction at every time step, targets = input shifted by one (Section 4.6). This is the Lab 13-14 architecture. |
| **Stateful** (Phase 5C) | Same LSTM sizes, stateful=True, fixed batch shape, ordered unshuffled batches, states reset each epoch (Section 4.6). Batch size must divide the sequence count - assert it. |
| **Char-CNN** (Phase 6) | Embedding (vocab x 256) - Conv1D blocks (e.g. 256 filters, kernel 5, ReLU, then kernel 3) with pooling / global pooling - Dense - Dropout - Dense(vocab, Softmax). Exact filter counts are the agent's design choice within this shape; record them in the config and results.json. |
| **Baselines** (Phase 3) | (a) Most-frequent next character; (b) character n-gram (order 3-5, agent's choice, recorded); (c) small vanilla (SimpleRNN) model. Baselines exist so the main model's numbers mean something. |

### 6.3 Two configurations - smoke first, full second

| Setting | FAST smoke config | FULL config (main model only) |
|---|---|---|
| Data | Subset of training tunes (about 10-20%) | Full training split |
| Epochs | About 5 | About 100, with best-checkpoint saving on validation loss and early-stopping patience recorded in config |
| Batch size | 16 | 16 (unless E11 changes it - then record why) |
| Sequence length | 64 | 64 |
| Purpose | Prove the whole pipeline end-to-end, cheaply | The reported main result |

![Planned compute budget chart showing smoke, experiment, variant and full run epochs](.src/media/budget.png)

Figure: planned epoch budgets by run type (design specification from Sections 6.3 and 8 - not measured results). Experiments never use the full budget; that is what keeps the build feasible on free Colab plus a laptop CPU.

## 7 Build phases 0-8

Each phase lists its tasks and its **exit criteria**. All exit criteria must pass and be logged before the next phase starts (Section 1.1).

### Phase 0 - Environment and dataset

**Tasks:** create the Section 5.1 tree; set up the local environment and a Colab session (5.2-5.3); download Nottingham (4.1); verify conversion tools on one tune.

**Exit criteria:** folder tree exists; TensorFlow import + GPU-visibility check logged (local: no GPU expected); dataset file count logged and 3 tunes inspected; one tune successfully converted abc -> MIDI -> WAV.

### Phase 1 - Exploratory data analysis

**Tasks:** compute tune count; corpus character count; vocabulary size; character frequency distribution; note-letter, meter (M:), and key (K:) distributions; tune-length distribution; plot 2-3 sample tunes (as text excerpts plus, if tooling allows, rendered notation or piano-roll from MIDI).

**Exit criteria:** EDA figures saved in outputs/figures/ (character distribution, meter distribution, key distribution, tune lengths); dataset statistics written to results.json; vocabulary size is in the expected small range (else return to cleaning rules).

### Phase 2 - Data preparation pipeline

**Tasks:** implement data_prep.py exactly per Sections 4.5-4.6: cleaning, tune split (seed 42, about 90/10), vocab files, many-to-one and many-to-many sequence builders, stateful ordering builder.

**Exit criteria (unit checks - all must assert true):** vocab round-trip (encode -> decode reproduces a sample tune); sequence shapes are (N, 64) inputs with scalar targets (many-to-one) and (N, 64) targets (many-to-many); no sequence crosses a tune boundary; no tune appears in both splits; stateful batch count is divisible by batch size. Check results logged.

### Phase 3 - Baselines

**Tasks:** implement and evaluate on the validation split: most-frequent next-char, character n-gram, and a small vanilla SimpleRNN. Record accuracy and perplexity for each in results.json.

**Exit criteria:** all three baselines have measured validation numbers saved. (The main model must later beat them; if it does not, that is a bug to fix, not a result to hide.)

### Phase 4 - Main Char-RNN training

**Tasks:** build the Section 6.1 model in model.py; run the FAST smoke config locally or on Colab and generate one sample tune from the smoke checkpoint; then run the FULL config on Colab with best-checkpoint saving.

**Exit criteria:** smoke run completes end-to-end including generation; full-run training history saved; training curves (loss, accuracy, perplexity - train vs validation) plotted to outputs/figures/; best checkpoint saved in outputs/checkpoints/; final train/val loss, accuracy, and perplexity written to results.json; main model beats all Phase 3 baselines on validation accuracy - or the shortfall is logged with a diagnosis.

### Phase 5 - Many-to-many and stateful variants

**Tasks:** build and train variant B (many-to-many, TimeDistributed Dense) and variant C (stateful) from Section 6.2, on the reduced experiment budget (Section 8), same split. Generate one sample tune from each.

**Exit criteria:** both variants train without shape/state errors; validation loss, accuracy, and perplexity recorded per variant; stateful run's log confirms shuffle=False and per-epoch state resets; sample tunes saved.

### Phase 6 - Char-CNN comparison

**Tasks:** build and train the Char-CNN (Section 6.2) on the same reduced budget and split; generate one sample tune.

**Exit criteria:** Char-CNN validation metrics recorded; architecture (filters, kernels) written to its config and to results.json; sample tune saved. Together with Phases 4-5 this completes the E10 comparison set.

### Phase 7 - Generation and conversion

**Tasks:**

1. Implement autoregressive generation in generate.py: seed with a valid header + opening fragment, sample one character at a time with temperature, feed it back, stop at a clean ending or a maximum length (set and record it, e.g. 600-1000 characters).
2. Generate at temperatures **0.6, 0.8, 1.0, 1.2** (this is also experiment E8).
3. Validate every generated tune with the Section 11.1 validator; apply only the documented repair rules; count tunes that remain invalid - never silently discard them from the validity rate.
4. Convert valid tunes with convert.py: abc -> MIDI -> WAV.
5. Save **at least 5** generated tunes as .abc + .mid + .wav in outputs/generated/, spanning different temperatures.
6. **Tabla extension (7.6):** produce one generation using a rhythm-focused seed / percussive pattern prompt and a short report note on what the abc model can and cannot represent about tabla - an honest extension demonstration (Section 3.1), not a claimed tabla model.

**Exit criteria:** 5+ triplets (.abc/.mid/.wav) exist and each .wav plays (non-silent, non-zero duration - assert programmatically); validity rate computed over *all* generation attempts in this phase and recorded; temperature set covers all four values.

### Phase 8 - Experiments, evaluation, and report assembly

**Tasks:** run the remaining experiments E1-E12 (Section 8 - several are already completed inside Phases 3-7; do not re-run those, reuse their recorded results); run the full evaluation protocol (Section 9); assemble the final report and viva sheet (Section 10.4); test the README commands from a clean run (Section 12).

**Exit criteria:** experiments table complete (CSV + report) with a status for every experiment including any honestly marked failed/not_run; results.json passes the Section 10.3 schema check; report and viva sheet exist; README clean-run test logged.

**Phase discipline reminder:** a phase that produces a number also produces the file that proves it - history, log, figure, or generated tune. If there is no file, the number does not go in the report.

## 8 Experiments E1-E12 - one variable at a time

**Protocol for every experiment:** change exactly the one named variable; keep data split, sequence length (except E7), model size (except E6), seed, and evaluation identical; run on the **reduced budget** (subset and/or about 10 epochs - record the exact budget used per experiment); record the outcome in outputs/results.json under experiments and in outputs/experiments.csv. Only the Phase 4 main model uses the full budget.

| ID | Experiment | The one variable changed | Recorded result |
|---|---|---|---|
| **E1** | Optimizer tournament | Optimizer only: SGD, SGD+Momentum, NAG, AdaGrad, AdaDelta, RMSProp, Adam | Val loss / accuracy / perplexity per optimizer + epochs-to-plateau note |
| **E2** | Gradient clipping | Clipping on vs off, at a deliberately high learning rate | Loss behaviour (stable vs spiking/diverged), final val metrics |
| **E3** | Dropout | Dropout rate 0 vs 0.3 vs 0.5 | Train-val gap + val metrics per rate |
| **E4** | BatchNorm variant | Add Batch Normalization (in the variant architecture) vs without | Val metrics + convergence speed note |
| **E5** | Activation / initialization | On a small MLP or RNN variant if feasible: activation / init scheme. If not feasible in budget, deliver as a written analysis tied to E2 + CNN ReLU evidence - and label it analysis, not experiment | Metrics if run; otherwise analysis section reference |
| **E6** | LSTM size | Units 128 vs 256 vs 512 | Val metrics + parameter count + time/epoch |
| **E7** | Sequence length | 32 vs 64 vs 100 characters | Val metrics + qualitative generation note |
| **E8** | Temperature study | Sampling temperature 0.6 / 0.8 / 1.0 / 1.2 (Phase 7 - reuse, do not re-run) | Validity rate + listening note per temperature |
| **E9** | Stateful vs stateless | Statefulness only (Phase 5 - reuse) | Val metrics + generation coherence note |
| **E10** | Model family comparison | Char-RNN vs Char-CNN vs many-to-many RNN (Phases 4-6 - reuse) | Val metrics side by side + sample tune per family |
| **E11** | Hyperparameter tuning | Small grid: learning rate x batch size (e.g. 3 x 2) | Metric per cell; best cell named and justified |
| **E12** | GPU vs CPU timing | Hardware only: identical config (same for local vs Colab) vs GPU) - fix E12 line | Seconds/epoch + total time, both environments named |

E12 definition: identical small config (same subset, epochs, batch size) timed on local CPU and on Colab GPU. Report both timings and both environment descriptions; do not compare a local full run against a Colab smoke run.

**Compute realism rule.** If Colab time runs out, reduce the experiment subset first, then epochs - never drop an experiment silently. A dropped experiment is recorded as not_run with the reason.

## 9 Evaluation protocol

### 9.1 Quantitative - model quality (held-out tunes only)

- **Next-character accuracy** on the validation split (whole held-out tunes, Section 4.6).
- **Loss and perplexity** (= exp(loss)) on the same split, reported together - perplexity without its loss is not acceptable.
- Report the same three numbers for every baseline and variant, so all comparisons share one protocol.

### 9.2 Quantitative - generation quality

- **Validity rate:** percentage of ALL generation attempts (state the denominator, minimum 20 attempts) whose abc passes every check: starts with X:, contains T:, M:, K: headers in order, contains bar lines, ends with a final bar / clean termination, and converts to MIDI without error.
- **Corpus statistics comparison:** note-letter distribution, meter distribution, and key distribution for generated tunes vs the training corpus, plotted side by side in outputs/figures/. State the distance qualitatively (close / shifted / collapsed to one key) from the plots.
- **Originality check:** for each delivered tune, confirm it is not an exact copy of any training tune (exact-match search over the corpus) and record the result.

### 9.3 Qualitative

- Listening notes per delivered tune and per temperature (E8): coherent / repetitive / wandering / gibberish, in plain words, dated and attributed to the listening session - not generated text.
- Failure gallery in the report: at least one invalid / gibberish generation shown and explained. A report with no failures shown is not credible for a generative model.

**No invented numbers - again, because this is where projects cheat.** Do not state any accuracy, perplexity, or validity range as an expected result, in the report, README, slides, or viva. Expected behaviour is qualitative only: loss and perplexity should decrease with training, accuracy should beat the baselines, and validity should improve with training and fall as temperature rises. The report contains only measured values from results.json.

## 10 Output contract - exact deliverables

All paths are relative to the repository root (Section 5.1). A deliverable that exists somewhere else does not count.

| Deliverable | Exact path / requirement |
|---|---|
| Trained main model | outputs/checkpoints/ - best checkpoint (weights + config that rebuilds the architecture) |
| Training curves | outputs/figures/ - loss, accuracy, perplexity (train vs val) for the main model |
| EDA figures | outputs/figures/ - character, meter, key, tune-length distributions |
| Experiments table | outputs/experiments.csv + the same table in the report; one row per run, with status |
| Generated tunes (5+) | outputs/generated/ - at least 5 tunes, each as .abc + .mid + .wav, spanning at least 3 different temperatures; filenames include temperature, e.g. gen_T0.8_01.abc |
| Results file | outputs/results.json - schema in 10.3, complete and honest |
| Phase log | outputs/phase_log.md + per-run logs in outputs/logs/ |
| Final report | report/ - contents in 10.4 |
| Viva sheet | Inside the report (final page) - mapping in 10.4 |
| README | README.md - commands in 10.5, tested from a clean run |
| Source code | src/ modules named in Section 5.1 + experiments/ configs; no notebook-only logic |

### 10.3 results.json schema (required keys)

```
{
  "project": "Music Generation Using Char-RNN",
  "seed": 42,
  "environment": {"python": "...", "tensorflow": "...",
    "local": {"os": "Windows", "gpu_visible_to_tf": false},
    "colab_gpu": {"name": "...", "used_for": ["full training"]}},
  "dataset": {"name": "Nottingham Music Database",
    "source_used": "...", "tunes_total": 0, "tunes_train": 0,
    "tunes_val": 0, "chars_total": 0, "vocab_size": 0},
  "main_model": {"config": {"seq_len": 64, "embedding_dim": 256,
      "lstm_units": [256,256,256], "dropout": 0.3,
      "batch_size": 16, "epochs_planned": 100, "epochs_run": 0},
    "train": {"loss": 0, "accuracy": 0, "perplexity": 0},
    "val":   {"loss": 0, "accuracy": 0, "perplexity": 0}},
  "baselines": {"most_frequent": {}, "ngram": {}, "simple_rnn": {}},
  "generation": {"attempts": 0, "valid": 0, "validity_rate": 0,
    "temperatures": [0.6,0.8,1.0,1.2], "delivered_tunes": []},
  "experiments": {"E1": {"status": "done|failed|not_run", "...": "..."}},
  "gpu_cpu_timing": {"local_cpu_sec_per_epoch": 0,
    "colab_gpu_sec_per_epoch": 0, "config_used": "..."}
}
```

Zeros above are schema placeholders, not results. The building agent replaces every one with a measured value or sets the block status to failed/not_run with a reason. A results.json still containing placeholder zeros at submission is a failed Definition of Done.

### 10.4 Final report - required contents

1. Problem statement and real-world framing (from Section 2)
2. Dataset: source, license, measured statistics, EDA figures, split and leakage rule
3. Architecture: main model, variants, Char-CNN, baselines - with the architecture diagram
4. Training: configs (smoke vs full), curves, checkpoint rule
5. Experiments E1-E12: table plus one paragraph per experiment - what changed, what happened, what it shows about the syllabus concept
6. Generation and evaluation: validity rate with denominator, corpus-statistics comparison, listening notes, failure gallery, tabla extension note
7. Limitations (Section 11 themes, as actually experienced) and conclusion
8. Dataset citation: Nottingham Music Database (public-domain folk tunes), cleaned version via jukedeck/nottingham-dataset; MAESTRO only if actually used
9. **Final page - Viva sheet:** a one-page table, "Where in my project is X?" - one row per syllabus topic from Section 3, pointing to the exact chapter, figure, experiment, or file. Build it from Section 3, do not write it from memory at the end.

### 10.5 README - exact run commands (zero ambiguity)

```
# 1. Setup (local)
python -m venv .venv
.venv\Scriptsctivate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
# 2. Data
python src/data_prep.py --download --split-seed 42
# 3. Train - smoke first, then full (full run: Colab GPU)
python src/train.py --config smoke
python src/train.py --config full
# 4. Generate + convert (at least 5 tunes)
python src/generate.py --checkpoint outputs/checkpoints/best --temperatures 0.6 0.8 1.0 1.2 --count 5
python src/convert.py --in outputs/generated --out outputs/generated
# 5. Evaluate (updates outputs/results.json + experiments.csv)
python src/evaluate.py --all
```

The agent may adjust argument names to its implementation, but the README must then document the actual commands - and Section 12 requires running them verbatim from a clean checkout.

## 11 Risks and traps

| Risk / trap | Mitigation written into this PRD |
|---|---|
| **Broken abc output** - missing headers, unmatched repeat signs, unterminated tunes; conversion fails | Validator + repair rules in 11.1. Count invalid tunes in the validity rate; never hand-edit delivered tunes into validity without recording the repair. |
| **Seed / temperature sensitivity** - one lucky seed looks like skill | Generate with multiple seeds; report validity over at least 20 attempts (9.2); deliver tunes across temperatures, not only the best one. |
| **Overfitting a small corpus** - about a thousand tunes is small for 3x LSTM 256 | Tune-level validation split, Dropout (E3), best-checkpoint on validation loss, train-val gap reported, E6 tests a smaller 128-unit model. |
| **Gibberish if undertrained** | Minimum guidance: do not judge generation from the smoke run. Generation deliverables come from the full-run checkpoint; if output is still incoherent at full epochs, first check data prep and vocab (Phase 2 checks), then training loss trend - in that order. |
| **Leakage by window split** | Split by tune before windowing (4.6); Phase 2 asserts no tune in both splits. |
| **TensorFlow Windows GPU trap** | Section 5: TF 2.11+ has no native-Windows GPU (last was 2.10). Train on Colab; E12 is local CPU vs Colab GPU by design. |
| **Paths with spaces / Windows paths** | Quote every path in commands; use forward slashes or escaped backslashes in configs; test README commands on the actual Windows laptop, not only on Colab/Linux. |
| **Colab session loss** | Download outputs/ at the end of every session (5.3); checkpoints first. |
| **MAESTRO audio download** | Backup is MIDI-only, about 57 MB. The audio version is about 87-122 GB - never download it (4.3). |
| **Stateful training silently shuffled** | shuffle=False asserted, ordering per 4.6, log confirmation required in Phase 5 exit criteria. |

### 11.1 abc validation and repair rules (generate.py)

1. **Validate:** starts with X:; T:, M:, K: present in order; at least one bar line; ends with a final bar (e.g. |] ) or a clean line end after a bar; balanced repeat signs (count of |: equals :| unless a repeat is opened and closed by the final bar convention used in the corpus - state the rule you implement); no characters outside the vocabulary.
2. **Repair (allowed, recorded):** prepend a standard header (X:, generated T:, M:/L:/K: taken from the seed) if the body is musical but the header is truncated; append a final bar if the tune ends mid-phrase at max length; drop a dangling unmatched repeat opener. Count every repaired tune separately: report raw validity and post-repair validity.
3. **Never repair** by replacing the body with a training tune, truncating to a few bars and calling it complete, or editing notes by hand. Those tunes stay invalid.

## 12 Definition of Done

- [ ] Phases 0-8 exit criteria all pass and are logged in phase_log.md
- [ ] results.json matches the 10.3 schema, contains only measured values, and has a status for E1-E12 (done / failed / not_run with reason)
- [ ] Main model beats all three baselines on validation accuracy, or the shortfall is diagnosed in the report
- [ ] At least 5 generated tunes delivered as .abc + .mid + .wav, playable, originality-checked, spanning at least 3 temperatures
- [ ] Validity rate reported with its denominator (20+ attempts), raw and post-repair
- [ ] Experiments table exists as CSV and in the report; training curves and EDA figures are generated from saved data
- [ ] Final report complete per 10.4, including limitations, failure gallery, and the one-page viva sheet
- [ ] README commands run verbatim, end to end, from a clean checkout / fresh environment, and the test is logged
- [ ] No fabricated number anywhere: report, README, and viva sheet all trace to results.json / logs / figures
- [ ] No paid or restricted dataset or service was used; dataset citation is present

## 13 Kickoff prompt - copy, paste, start Phase 0

Paste the block below to the coding agent (OpenCode / Muse), with this PRD available to it as a file or in context. It starts Phase 0 only - on purpose.

```
Read the PRD titled "Music Generation Using Char-RNN - PRD"
(Deep Learning Applications, 22PC1DS402 + 22PC2DS402) in full
before writing any code.

Follow its operating rules exactly: work phase by phase, run
each phase's exit criteria before moving on, never fabricate
any metric, and save every result to the output-contract
paths, updating outputs/results.json after every run.

Start with PHASE 0 ONLY: create the repository tree in
Section 5.1, set up the environment, download and verify
the Nottingham dataset per Section 4.1, and test
abc -> MIDI -> WAV conversion on one tune. Remember the
TensorFlow Windows GPU trap in Section 5 - local is CPU,
GPU training is Colab.

When Phase 0's exit criteria pass, show me the phase log
and stop for my confirmation before Phase 1.
```

**After kickoff:** at each phase gate, check the phase log yourself before confirming the next phase. The gates are the project - an agent that skips them will hand you a report you cannot defend in viva.

Sources for dataset facts used in this PRD: Nottingham Music Database via abc.sourceforge.net/NMD and ifdo.ca/~seymour/nottingham; cleaned version at github.com/jukedeck/nottingham-dataset (about 1,033 tunes in the cleaned set). Syllabus topic names are taken from the student's 22PC1DS402 / 22PC2DS402 syllabus documents. No measured results are cited in this PRD - all results are produced by the build.
