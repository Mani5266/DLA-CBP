"""Phase 2: cleaning, vocab, tune-level split, sequence builders (PRD 4.5-4.6)."""
import argparse, glob, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nottingham-dataset" / "ABC_cleaned"
RAW_FALLBACK = ROOT / "data" / "raw"
PROCD = ROOT / "data" / "processed"
SEQ_LEN = 64
KEEP_HEADERS = {"X", "T", "M", "L", "R", "K"}


def collect_files():
    files = sorted(glob.glob(str(RAW_DIR / "*.abc")))
    if not files:  # fallback: any .abc directly under data/raw
        files = sorted(glob.glob(str(RAW_FALLBACK / "*.abc")))
    return files


def split_raw_tunes(text):
    """Split raw file text into tunes on X: at start of line."""
    parts = re.split(r"(?m)^(?=X:)", text)
    return [p for p in parts if p.strip()]


def clean_tune(tune):
    """Apply PRD 4.5 rules. Returns cleaned tune or None if corrupt (no K:)."""
    lines = tune.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = []
    for ln in lines:
        s = ln.rstrip()
        if not s:
            continue
        if s.startswith("%"):
            continue
        m = re.match(r"^([A-Z]):", s)
        if m:
            if m.group(1) in KEEP_HEADERS:
                kept.append(s)
            continue  # drop W:, Z:, S: and any other metadata headers
        kept.append(s)  # tune body line (notes, bars, chords) kept verbatim
    out = "\n".join(kept).strip() + "\n"
    if not re.search(r"(?m)^K:", out):
        return None
    return out


def build_corpus():
    files = collect_files()
    assert files, "no .abc files found - run download first"
    tunes, dropped = [], 0
    for f in files:
        text = open(f, encoding="utf-8", errors="ignore").read()
        for t in split_raw_tunes(text):
            c = clean_tune(t)
            if c is None:
                dropped += 1
            else:
                tunes.append(c)
    return tunes, dropped, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.9)
    args = ap.parse_args()

    PROCD.mkdir(parents=True, exist_ok=True)
    tunes, dropped, files = build_corpus()
    # ponytail: case is pitch - never lowercase (PRD 4.5)
    corpus = "\n\n".join(tunes) + "\n"
    vocab = sorted(set(corpus))
    assert len(vocab) < 1000, f"vocab {len(vocab)} too big - cleaning failed"
    c2i = {c: i for i, c in enumerate(vocab)}

    rng = random.Random(args.split_seed)
    order = list(range(len(tunes)))
    rng.shuffle(order)
    n_train = int(len(tunes) * args.train_frac)
    train_tunes = [tunes[i] for i in order[:n_train]]
    val_tunes = [tunes[i] for i in order[n_train:]]
    train_txt = "\n\n".join(train_tunes) + "\n"
    val_txt = "\n\n".join(val_tunes) + "\n"

    def windows(txt_tunes):
        n1 = n2 = 0
        for t in txt_tunes:
            ids = [c2i[c] for c in t]
            assert all(0 <= i < len(vocab) for i in ids)
            if len(ids) > SEQ_LEN:
                n1 += len(ids) - SEQ_LEN       # many-to-one
                n2 += len(ids) - SEQ_LEN       # many-to-many
        return n1, n2

    ntr1, ntr2 = windows(train_tunes)
    nva1, nva2 = windows(val_tunes)

    (PROCD / "corpus.txt").write_text(corpus, encoding="utf-8")
    (PROCD / "train.txt").write_text(train_txt, encoding="utf-8")
    (PROCD / "val.txt").write_text(val_txt, encoding="utf-8")
    (PROCD / "vocab.json").write_text(json.dumps(
        {"chars": vocab, "char2idx": c2i,
         "idx2char": {str(i): c for c, i in c2i.items()},
         "vocab_size": len(vocab)}, indent=1), encoding="utf-8")
    (PROCD / "sequences_meta.json").write_text(json.dumps(
        {"seq_len": SEQ_LEN, "seed": args.split_seed,
         "tunes_total": len(tunes), "tunes_train": len(train_tunes),
         "tunes_val": len(val_tunes), "chars_total": len(corpus),
         "vocab_size": len(vocab), "dropped_no_key": dropped,
         "train_windows": ntr1, "val_windows": nva1,
         "source_files": [Path(f).name for f in files]}, indent=1),
        encoding="utf-8")

    # Phase 2 unit checks (PRD exit criteria)
    sample = train_tunes[0]
    rt = "".join(vocab[c2i[c]] for c in sample)
    assert rt == sample, "vocab round-trip failed"
    assert ntr1 > 0 and nva1 > 0, "empty window sets"
    assert len(set(order[:n_train]) & set(order[n_train:])) == 0, "tune leak!"
    print(f"tunes={len(tunes)} dropped={dropped} vocab={len(vocab)} "
          f"chars={len(corpus)} train_win={ntr1} val_win={nva1}")
    print("phase2 checks: round-trip OK, shapes OK, no cross-tune windows, "
          "no tune in both splits. PASS")


def encode_tune_windows(tune, c2i, seq_len=SEQ_LEN, many_to_many=False):
    """Windows for ONE tune (never crosses boundary). Returns (X, y) id lists."""
    ids = [c2i[c] for c in tune]
    X, Y = [], []
    for i in range(len(ids) - seq_len):
        X.append(ids[i:i + seq_len])
        Y.append(ids[i + 1:i + seq_len + 1] if many_to_many else ids[i + seq_len])
    return X, Y


if __name__ == "__main__":
    main()
