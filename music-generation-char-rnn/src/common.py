"""Shared paths, vocab loading, results.json helpers. (ponytail: one tiny
module beats the same 15 lines pasted into 4 scripts.)"""
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCD = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
RES = OUT / "results.json"


def load_vocab():
    v = json.loads((PROCD / "vocab.json").read_text(encoding="utf-8"))
    return v["chars"], v["char2idx"], v["vocab_size"]


def load_results():
    if RES.exists():
        return json.loads(RES.read_text(encoding="utf-8"))
    return {"project": "Music Generation Using Char-RNN", "seed": 42,
            "environment": {}, "dataset": {}, "main_model": {},
            "baselines": {}, "generation": {}, "experiments": {},
            "gpu_cpu_timing": {}}


def save_results(r):
    OUT.mkdir(parents=True, exist_ok=True)
    RES.write_text(json.dumps(r, indent=1), encoding="utf-8")


class Timer:
    def __init__(self):
        self.t = time.time()
    def lap(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        return dt
