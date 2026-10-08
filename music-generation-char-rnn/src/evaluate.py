"""Evaluation: --eda (Phase 1), --baselines (Phase 3), --final (Phase 8).
All plots from saved data; all numbers -> results.json + experiments.csv."""
import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, PROCD, OUT, load_vocab, load_results, save_results


def tune_list(path):
    return [t for t in Path(path).read_text(encoding="utf-8").split("\n\n")
            if t.strip()]


def field(tunes, f):
    c = Counter()
    for t in tunes:
        m = re.search(rf"(?m)^{f}:(.*)$", t)
        if m:
            c[m.group(1).strip().split()[0]] += 1
    return c


def dist_plot(counter, title, path, top=20):
    items = counter.most_common(top)
    plt.figure(figsize=(8, 4))
    plt.bar([k for k, _ in items], [v for _, v in items])
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def do_eda():
    from datetime import date
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    tunes = tune_list(PROCD / "corpus.txt")
    tr, va = tune_list(PROCD / "train.txt"), tune_list(PROCD / "val.txt")
    corpus = (PROCD / "corpus.txt").read_text(encoding="utf-8")
    chars, c2i, V = load_vocab()
    freq = Counter(corpus)
    dist_plot(freq, "Character frequency (cleaned corpus)",
              figs / "char_dist.png", top=30)
    dist_plot(field(tunes, "M"), "Meter distribution (M:)",
              figs / "meter_dist.png")
    dist_plot(field(tunes, "K"), "Key distribution (K:)",
              figs / "key_dist.png")
    lens = [len(t) for t in tunes]
    plt.figure(figsize=(8, 4))
    plt.hist(lens, bins=40)
    plt.title("Tune length distribution (chars)")
    plt.tight_layout()
    plt.savefig(figs / "tune_lengths.png")
    plt.close()
    notes = Counter(re.findall(r"[A-Ga-g]", corpus))
    r = load_results()
    r["dataset"] = {"name": "Nottingham Music Database",
        "source_used": "github.com/jukedeck/nottingham-dataset (cleaned)",
        "license": "public domain folk tunes",
        "tunes_total": len(tunes), "tunes_train": len(tr),
        "tunes_val": len(va), "chars_total": len(corpus),
        "vocab_size": V, "vocab": "".join(chars),
        "top_chars": freq.most_common(10),
        "meters": dict(field(tunes, "M").most_common(10)),
        "keys": dict(field(tunes, "K").most_common(10)),
        "note_letters": dict(notes),
        "mean_tune_len": float(np.mean(lens)),
        "eda_date": str(date.today())}
    save_results(r)
    print(f"EDA: tunes={len(tunes)} chars={len(corpus)} vocab={V} "
          f"meters={dict(field(tunes,'M').most_common(5))}")
    print("EDA PASS")


def do_baselines():
    chars, c2i, V = load_vocab()
    tr = tune_list(PROCD / "train.txt")
    va = tune_list(PROCD / "val.txt")
    tri = [[c2i[c] for c in t if c in c2i] for t in tr]
    vai = [[c2i[c] for c in t if c in c2i] for t in va]
    flat = [i for t in tri for i in t]
    uni = Counter(flat)
    top = uni.most_common(1)[0][0]
    # most-frequent: 0.99 on top char, rest uniform (stated rule -> finite loss)
    tot = corr = loss = 0.0
    for t in vai:
        for j in range(64, len(t)):
            tot += 1
            corr += t[j] == top
            loss += -math.log(0.99 if t[j] == top else 0.01 / (V - 1))
    mf = {"accuracy": corr / tot, "loss": loss / tot,
          "perplexity": math.exp(loss / tot)}
    # char 4-gram with add-1 smoothing over vocab
    N = 4
    ctx_counts, ctx_tot = {}, Counter()
    for t in tri:
        for j in range(N - 1, len(t)):
            k = tuple(t[j - N + 1:j])
            ctx_counts[k] = ctx_counts.get(k, Counter())
            ctx_counts[k][t[j]] += 1
            ctx_tot[k] += 1
    tot = corr = loss = 0.0
    for t in vai:
        for j in range(N - 1, len(t)):
            k = tuple(t[j - N + 1:j])
            tot += 1
            cc = ctx_counts.get(k, Counter())
            denom = ctx_tot.get(k, 0) + V
            pred = max(cc, key=lambda c: cc[c]) if cc else top
            corr += t[j] == pred
            loss += -math.log((cc.get(t[j], 0) + 1) / denom)
    ng = {"accuracy": corr / tot, "loss": loss / tot,
          "perplexity": math.exp(loss / tot), "order": N}
    r = load_results()
    r["baselines"] = {"most_frequent": {**mf, "status": "done"},
                      "ngram": {**ng, "status": "done"},
                      "simple_rnn": r.get("baselines", {}).get(
                          "simple_rnn", {"status": "not_run"})}
    save_results(r)
    print(f"baselines: most_freq acc={mf['accuracy']:.3f} ppl={mf['perplexity']:.1f} | "
          f"4-gram acc={ng['accuracy']:.3f} ppl={ng['perplexity']:.1f}. PASS")


def do_curves(tag="main_full"):
    figs = OUT / "figures"
    h = json.loads((OUT / "logs" / f"history_{tag}.json").read_text())
    ep = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(ep, h["loss"], label="train")
    ax[0].plot(ep, h["val_loss"], label="val")
    ax[0].set_title("Loss (cross-entropy)")
    ax[0].legend()
    ax[1].plot(ep, h["acc"], label="train")
    ax[1].plot(ep, h["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[2].plot(ep, h["ppl"], label="train")
    ax[2].plot(ep, h["val_ppl"], label="val")
    ax[2].set_title("Perplexity = exp(loss)")
    ax[2].legend()
    fig.suptitle(f"Training curves ({tag}) - from saved history, not hand-drawn")
    fig.tight_layout()
    fig.savefig(figs / f"curves_{tag}.png")
    plt.close(fig)
    print(f"curves_{tag}.png from {len(ep)} epochs")


def do_final():
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    r = load_results()
    ex = r.get("experiments", {})
    if "baseline_small_rnn" in ex:  # sync neural baseline into baselines block
        sm = ex["baseline_small_rnn"]
        r["baselines"]["simple_rnn"] = {**sm.get("val", {}), "status": "done",
                                        "params": sm.get("config", {}).get(
                                            "params")}
        save_results(r)
    gen_dir = OUT / "generated"
    tr = tune_list(PROCD / "train.txt")
    tr_set = {"".join(t.split()) for t in tr}
    gen_files = sorted(gen_dir.glob("*.abc"))
    orig_ok = 0
    NK, MK, KK = Counter(), Counter(), Counter()
    for f in gen_files:
        t = f.read_text(encoding="utf-8")
        if "".join(t.split()) not in tr_set:
            orig_ok += 1
        NK.update(re.findall(r"[A-Ga-g]", t))
        m = re.search(r"(?m)^M:(.*)$", t)
        k = re.search(r"(?m)^K:(.*)$", t)
        if m:
            MK[m.group(1).strip().split()[0]] += 1
        if k:
            KK[k.group(1).strip().split()[0]] += 1
    corp_notes = r["dataset"].get("note_letters", {})
    if corp_notes and NK:
        keys = sorted(set(corp_notes) | set(NK))
        c = np.array([corp_notes.get(k, 0) for k in keys], float)
        g = np.array([NK.get(k, 0) for k in keys], float)
        c, g = c / c.sum(), g / g.sum()
        x = np.arange(len(keys))
        plt.figure(figsize=(9, 4))
        plt.bar(x - 0.2, c, 0.4, label="corpus")
        plt.bar(x + 0.2, g, 0.4, label="generated")
        plt.xticks(x, keys)
        plt.title("Note-letter distribution: corpus vs generated")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs / "note_compare.png")
        plt.close()
    g = r.get("generation", {})
    g["originality"] = {"checked": len(gen_files), "original": orig_ok}
    save_results(r)
    # experiments.csv from results.json (one row per run, with status)
    rows = []
    mm = r.get("main_model", {})
    if mm:
        rows.append(("main_full", "done", mm.get("val", {})))
    for k, v in r.get("baselines", {}).items():
        rows.append((f"baseline_{k}", v.get("status", "done"), v))
    for k, v in r.get("experiments", {}).items():
        rows.append((k, v.get("status", "done"),
                     v.get("val", v) if isinstance(v, dict) else v))
    with open(OUT / "experiments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "status", "loss", "accuracy", "perplexity"])
        for name, st, m in rows:
            w.writerow([name, st, round(m.get("loss", 0), 4)
                        if isinstance(m, dict) else m,
                        round(m.get("accuracy", 0), 4)
                        if isinstance(m, dict) else "",
                        round(m.get("perplexity", 0), 2)
                        if isinstance(m, dict) else ""])
    print(f"final: {len(gen_files)} tunes checked, {orig_ok} original; "
          f"csv rows={len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--curves", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.eda or a.all:
        do_eda()
    if a.baselines or a.all:
        do_baselines()
    if a.final or a.all:
        do_final()
    if a.curves:
        do_curves(a.curves)
