"""Config-driven training (PRD 6.3, 8). Usage:
python src/train.py --config smoke
python src/train.py --config full --tag main_full
python src/train.py --config exp --tag E1_adam --opt adam --epochs 8 --subset 0.15
"""
import argparse, json, math, sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, PROCD, OUT, load_vocab, load_results, save_results, Timer
from model import build, count_params

PROCD_TRAIN = PROCD / "train.txt"
PROCD_VAL = PROCD / "val.txt"

PRESETS = {
    "smoke": dict(subset=0.15, epochs=5, batch=64, seq_len=64, stride=1,
                   lr=3e-3, opt="adam", clip=1.0, dropout=0.3,
                   units=(256, 256, 256), model="main", patience=99),
    "full": dict(subset=1.0, epochs=60, batch=128, seq_len=64, stride=2,
                  lr=3e-3, opt="adam", clip=1.0, dropout=0.3,
                  units=(256, 256, 256), model="main", patience=8),
    "exp": dict(subset=0.15, epochs=8, batch=64, seq_len=64, stride=2,
                 lr=3e-3, opt="adam", clip=1.0, dropout=0.3,
                 units=(256, 256, 256), model="main", patience=99),
}


def get_opt(name, params, lr):
    n = name.lower()
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if n == "sgd_mom":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if n == "nag":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if n == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if n == "adadelta":
        return torch.optim.Adadelta(params, lr=lr)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def load_windows(path, c2i, seq_len, stride, many, frac, seed):
    tunes = [t for t in path.read_text(encoding="utf-8").split("\n\n") if t.strip()]
    rng_state = hash((seed, str(path))) & 0xFFFF
    import random
    tunes = tunes[:max(1, int(len(tunes) * frac))]
    X, Y = [], []
    for t in tunes:
        ids = [c2i[c] for c in t if c in c2i]
        for i in range(0, len(ids) - seq_len, stride):
            X.append(ids[i:i + seq_len])
            Y.append(ids[i + 1:i + seq_len + 1] if many else ids[i + seq_len])
    return torch.tensor(X, dtype=torch.long), (
        torch.tensor(Y, dtype=torch.long) if not many
        else torch.tensor(Y, dtype=torch.long)), tunes


def evaluate(model, loader, device, many):
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, tot_acc, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = ce(out.view(-1, out.shape[-1]), yb.view(-1))
            pred = out.argmax(-1)
            tot_loss += loss.item() * len(xb)
            tot_acc += (pred.view(-1) == yb.view(-1)).float().mean().item() * len(xb)
            n += len(xb)
    l = tot_loss / max(n, 1)
    a = tot_acc / max(n, 1)
    return l, a, math.exp(min(l, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--opt", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--units", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batchnorm", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--patience", type=int, default=None)
    a = ap.parse_args()

    cfg = dict(PRESETS[a.config])
    for k in ("opt", "lr", "clip", "dropout", "seq_len", "batch", "epochs",
              "subset", "stride", "model", "patience"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.units:
        cfg["units"] = (a.units, a.units, a.units)
    if a.batchnorm:
        cfg["batchnorm"] = True
    tag = a.tag or (a.config if a.config in ("smoke", "full") else "exp")
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available()
                          else "cuda")
    many = cfg["model"] == "many2many"
    stateful = cfg["model"] == "stateful"

    chars, c2i, V = load_vocab()
    Xtr, ytr, tr_tunes = load_windows(PROCD_TRAIN, c2i, cfg["seq_len"],
                                      cfg["stride"], many, cfg["subset"], 42)
    Xva, yva, va_tunes = load_windows(PROCD_VAL, c2i, cfg["seq_len"],
                                      cfg["stride"], many, 1.0, 42)
    if stateful:  # trim so batch divides count; ordered, no shuffle (PRD 4.6)
        n = (len(Xtr) // cfg["batch"]) * cfg["batch"]
        Xtr, ytr = Xtr[:n], ytr[:n]
        assert len(Xtr) % cfg["batch"] == 0
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg["batch"],
                           shuffle=not stateful)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg["batch"])

    kw = dict(emb=256, dropout=cfg["dropout"],
              batchnorm=cfg.get("batchnorm", False))
    if cfg["model"] == "cnn":
        kw.update(filters=(256, 256), kernels=(5, 3))
    elif cfg["model"] == "small_rnn":
        kw.update(emb=64, hidden=128)
    else:
        kw["units"] = cfg["units"]
    model = build(cfg["model"], V, **kw).to(device)
    n_params = count_params(model)
    opt = get_opt(cfg["opt"], model.parameters(), cfg["lr"])
    ce = nn.CrossEntropyLoss()

    hist = {"loss": [], "acc": [], "ppl": [], "val_loss": [], "val_acc": [],
            "val_ppl": [], "sec_per_epoch": []}
    best, bad, best_state = 1e9, 0, None
    print(f"[{tag}] device={device} params={n_params} "
          f"train_win={len(Xtr)} val_win={len(Xva)} cfg={cfg}", flush=True)
    for ep in range(cfg["epochs"]):
        t = Timer()
        model.train()
        h = None
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if stateful:
                out, h = model(xb, h, return_state=True)
                h = tuple((hh.detach(), ch.detach()) for hh, ch in h)
            else:
                out = model(xb)
            loss = ce(out.view(-1, out.shape[-1]), yb.view(-1))
            loss.backward()
            if cfg["clip"] and cfg["clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
        h = None  # reset states each epoch (PRD 4.6)
        tr_l, tr_a, tr_p = evaluate(model, tr_loader, device, many)
        va_l, va_a, va_p = evaluate(model, va_loader, device, many)
        dt = t.lap()
        for k, v in zip(("loss", "acc", "ppl", "val_loss", "val_acc",
                         "val_ppl"), (tr_l, tr_a, tr_p, va_l, va_a, va_p)):
            hist[k].append(v)
        hist["sec_per_epoch"].append(dt)
        print(f"[{tag}] ep{ep+1}/{cfg['epochs']} {dt:.0f}s "
              f"loss={tr_l:.3f} acc={tr_a:.3f} ppl={tr_p:.1f} | "
              f"val_loss={va_l:.3f} val_acc={va_a:.3f} val_ppl={va_p:.1f}",
              flush=True)
        if va_l < best:
            best, bad = va_l, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[{tag}] early stop at ep{ep+1}", flush=True)
                break

    ckpt = OUT / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "config": {**cfg, "vocab_size": V},
                "val_loss": best}, ckpt)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    hist_f = OUT / "logs" / f"history_{tag}.json"
    hist_f.write_text(json.dumps(hist, indent=1), encoding="utf-8")

    r = load_results()
    import platform
    r["environment"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "local": {"os": platform.system(),
                  "gpu_visible": torch.cuda.is_available(),
                  "gpu_name": torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else None},
        "note": "PyTorch+CUDA used: TF 2.11+ has no native-Windows GPU "
                "(last GPU build TF 2.10, py<=3.10); same Char-RNN maths."}
    entry = {"status": "done", "config": {**cfg, "params": n_params},
             "epochs_run": len(hist["loss"]),
             "train": {"loss": hist["loss"][-1], "accuracy": hist["acc"][-1],
                       "perplexity": hist["ppl"][-1]},
             "val": {"loss": hist["val_loss"][-1],
                     "accuracy": hist["val_acc"][-1],
                     "perplexity": hist["val_ppl"][-1]},
             "sec_per_epoch": sum(hist["sec_per_epoch"]) /
             max(len(hist["sec_per_epoch"]), 1),
             "checkpoint": str(ckpt)}
    if tag in ("full", "main_full"):
        mc = entry["config"]
        r["main_model"] = {"config": {"seq_len": mc["seq_len"],
            "embedding_dim": 256, "lstm_units": list(mc["units"]),
            "dropout": mc["dropout"], "batch_size": mc["batch"],
            "epochs_planned": mc["epochs"], "epochs_run": entry["epochs_run"],
            "stride": mc["stride"], "optimizer": mc["opt"], "lr": mc["lr"]},
            "train": entry["train"], "val": entry["val"]}
    else:
        r.setdefault("experiments", {})[tag] = entry
    save_results(r)
    print(f"[{tag}] saved {ckpt} + results.json. DONE")


if __name__ == "__main__":
    sys.exit(main())
