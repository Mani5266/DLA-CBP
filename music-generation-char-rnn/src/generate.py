"""Autoregressive generation with temperature + abc validator/repair (PRD 7, 11.1)."""
import argparse, json, random, re
from pathlib import Path

import torch

from common import ROOT, PROCD, OUT, load_vocab, load_results, save_results
from model import build

MAX_LEN = 800


def rebuild(cfg, V):
    kw = dict(emb=256, dropout=cfg.get("dropout", 0.3),
              batchnorm=cfg.get("batchnorm", False))
    name = cfg.get("model", "main")
    if name == "cnn":
        kw.update(filters=(256, 256), kernels=(5, 3))
    elif name == "small_rnn":
        kw.update(emb=64, hidden=128)
    else:
        kw["units"] = tuple(cfg.get("units", (256, 256, 256)))
    return build(name, V, **kw)


def validate(tune, vocab_set):
    """PRD 11.1 checks. Returns (ok, reasons list)."""
    reasons = []
    if not tune.startswith("X:"):
        reasons.append("missing X:")
    ix = tune.find("X:")
    it, im, ik = tune.find("\nT:"), tune.find("\nM:"), tune.find("\nK:")
    if not (it > ix and im > it and ik > im):
        reasons.append("T/M/K headers missing or out of order")
    if "|" not in tune:
        reasons.append("no bar lines")
    tail = tune.rstrip()
    if not (tail.endswith("|]") or tail.endswith("|") or tail.endswith(":|")):
        reasons.append("no clean bar ending")
    # corpus rule (measured 930/930 train tunes): closers >= openers
    # (repeat-from-start); only a dangling OPENER is invalid -> repaired
    if tune.count("|:") > tune.count(":|"):
        reasons.append("unbalanced repeats")
    bad = set(tune) - vocab_set
    if bad:
        reasons.append(f"chars outside vocab: {sorted(bad)[:5]}")
    return len(reasons) == 0, reasons


def repair(tune, seed_header):
    """Only the allowed repairs (PRD 11.1). Returns (tune, notes)."""
    notes = []
    if not tune.startswith("X:"):
        tune = seed_header + tune.lstrip()  # body musical, header truncated
        notes.append("prepended header")
    if tune.count("|:") > tune.count(":|"):
        tune = tune.replace("|:", "|", tune.count("|:") - tune.count(":|"))
        notes.append("dropped dangling repeat opener")
    tail = tune.rstrip()
    if not (tail.endswith("|]") or tail.endswith("|") or tail.endswith(":|")):
        tune = tail + "|]\n"  # ended mid-phrase at max length
        notes.append("appended final bar")
    return tune, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--attempts-per-temp", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="gen")
    ap.add_argument("--no-record", action="store_true",
                    help="save tunes but do not touch results.json counters")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    chars, c2i, V = load_vocab()
    vocab_set = set(chars)
    i2c = {i: c for c, i in c2i.items()}
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rebuild(cfg, V)
    model.load_state_dict({k: v for k, v in ckpt["state"].items()})
    model.to(device)

    train_tunes = [t.strip() for t in (PROCD / "train.txt").read_text(
        encoding="utf-8").split("\n\n") if t.strip()]
    rng = random.Random(a.seed)
    gdir = OUT / "generated"
    gdir.mkdir(parents=True, exist_ok=True)

    attempts = valid_raw = valid_repaired = 0
    delivered, per_temp = [], {}
    for temp in a.temperatures:
        per_temp[str(temp)] = {"attempts": 0, "valid": 0}
        for k in range(a.attempts_per_temp):
            src = rng.choice(train_tunes)
            cut = rng.randint(40, min(110, len(src) - 20))
            seed_txt = src[:cut]
            seed_header = seed_txt[:seed_txt.find("\nK:")]
            seed_header = seed_header[:seed_header.rfind("\n") + 1]
            if "K:" not in seed_header:  # keep full header always
                seed_header = src[:src.find("\n", src.find("K:")) + 1]
                seed_txt = seed_header + src[len(seed_header):cut]
            seed_ids = [c2i[c] for c in seed_txt[-64:]]
            ids = list([c2i[c] for c in seed_txt])
            # generate until final bar or MAX_LEN
            cur = list(ids)
            with torch.no_grad():
                while len(cur) < MAX_LEN:
                    x = torch.tensor([cur[-64:]], dtype=torch.long).to(device)
                    out = model(x).cpu().squeeze(0)
                    if out.dim() == 2:  # many-to-many variant: use last step
                        out = out[-1]
                    nxt = torch.multinomial(torch.softmax(
                        out / max(temp, 1e-6), -1), 1).item()
                    cur.append(nxt)
                    # corpus tunes end with :| (repeat) or | - stop there
                    if len(cur) > 120 and i2c[cur[-2]] == ":" and \
                            i2c[cur[-1]] == "|":
                        break
                    if len(cur) > 120 and i2c[cur[-2]] == "|" and \
                            i2c[cur[-1]] == "]":
                        break
            tune = "".join(i2c[i] for i in cur)
            attempts += 1
            per_temp[str(temp)]["attempts"] += 1
            ok, _ = validate(tune, vocab_set)
            status = "raw" if ok else "invalid"
            if ok:
                valid_raw += 1
            else:
                tune, notes = repair(tune, seed_header)
                ok2, _ = validate(tune, vocab_set)
                if ok2:
                    valid_repaired += 1
                    status = "repaired:" + "+".join(notes)
            if status != "invalid":
                per_temp[str(temp)]["valid"] += 1
                fname = f"{a.tag}_T{temp}_{k}.abc"
                (gdir / fname).write_text(tune, encoding="utf-8")
                delivered.append({"file": fname, "temperature": temp,
                                  "status": status, "chars": len(tune)})
            print(f"T={temp} try{k} {status} len={len(tune)}", flush=True)

    r = load_results()
    g = r.get("generation", {})
    if a.no_record:
        g["variant_samples"] = g.get("variant_samples", []) + delivered
        save_results(r)
        print(f"saved {len(delivered)} variant sample(s), counters untouched")
        return
    g.update({"attempts": g.get("attempts", 0) + attempts,
              "valid": g.get("valid", 0) + valid_raw + valid_repaired,
              "valid_raw": g.get("valid_raw", 0) + valid_raw,
              "valid_repaired": g.get("valid_repaired", 0) + valid_repaired,
              "validity_rate": (g.get("valid", 0) + valid_raw + valid_repaired)
              / max(g.get("attempts", 0) + attempts, 1),
              "temperatures": a.temperatures,
              "delivered_tunes": g.get("delivered_tunes", []) + delivered,
              "per_temperature": per_temp, "max_len": MAX_LEN})
    save_results(r)
    print(f"attempts={attempts} raw={valid_raw} repaired={valid_repaired} "
          f"rate={(valid_raw+valid_repaired)/max(attempts,1):.2f}")


if __name__ == "__main__":
    main()
