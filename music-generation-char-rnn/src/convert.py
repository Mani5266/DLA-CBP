"""abc -> MIDI (mido) -> WAV (numpy sine synth). (ponytail: pure-python, no
TiMidity/soundfont install; report states sine-render honestly.)"""
import argparse, re
from pathlib import Path

import mido
import numpy as np

SR = 22050
BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# key-signature sharps/flats for common folk keys (major/minor tonic)
KEY_ACC = {"G": {"F": 1}, "D": {"F": 1, "C": 1}, "A": {"F": 1, "C": 1, "G": 1},
           "F": {"B": -1}, "Bb": {"B": -1, "E": -1}, "Eb": {"B": -1, "E": -1,
           "A": -1}, "C": {}, "Am": {}, "Em": {"F": 1}, "Dm": {"B": -1},
           "Bm": {"F": 1, "C": 1}, "Gm": {"B": -1, "E": -1}}


def header(abc, field):
    m = re.search(rf"(?m)^{field}:(.*)$", abc)
    return m.group(1).strip() if m else ""


def frac(s, default=1.0):
    if not s:
        return default
    if "/" in s:
        a, _, b = s.partition("/")
        return (float(a) if a else 1.0) / (float(b) if b else 2.0)
    return float(s)


def parse_abc(abc):
    """Minimal single-melody parser. Returns [(midi|None(rest), beats)]."""
    l_def = frac(header(abc, "L") or "1/8")
    key = header(abc, "K").split()[0] if header(abc, "K") else "C"
    sig = KEY_ACC.get(key, KEY_ACC.get(key.rstrip("m"), {}))
    q_beats = l_def * 4  # quarter-note = 1 beat
    body = "\n".join(l for l in abc.split("\n")
                     if not re.match(r"^[A-Z]:", l))
    body = re.sub(r'"[^"]*"', "", body)  # chord symbols
    body = re.sub(r"![^!]*!", "", body)
    body = body.replace(">", "").replace("<", "")
    toks = re.findall(r"(\^+|_+|=)?([A-Ga-gzZx])([,']*)(\d*/?\d*)", body)
    out = []
    for acc, note, octv, dur in toks:
        beats = (frac(dur) if dur else 1.0) * q_beats
        if note in "zZx":
            out.append((None, beats))
            continue
        up = note.upper()
        semi = BASE[up]
        if acc:
            semi += 1 if "^" in acc else (-1 if "_" in acc else 0)
            if "=" in acc:
                semi = BASE[up]
        elif up in sig:
            semi += sig[up]
        midi = 60 + semi + (12 if note.islower() else 0)
        midi += 12 * octv.count("'") - 12 * octv.count(",")
        out.append((midi, beats))
    return out, key


def to_midi(notes, path, tempo=500000):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.Message("program_change", program=0, time=0))
    for midi, beats in notes:
        ticks = max(20, int(beats * 480))
        if midi is None:
            tr.append(mido.Message("note_on", note=60, velocity=0, time=ticks))
        else:
            tr.append(mido.Message("note_on", note=midi, velocity=90, time=0))
            tr.append(mido.Message("note_off", note=midi, velocity=64,
                                   time=ticks))
    mid.save(path)
    return path


def midi_to_wav(mid_path, wav_path):
    from scipy.io import wavfile
    mid = mido.MidiFile(mid_path)
    tpq, tempo = mid.ticks_per_beat, 500000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    sec_per_tick = tempo / 1e6 / tpq
    events = []  # (start_sec, midi, dur_sec)
    abs_tick, live = 0, {}
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            live[msg.note] = abs_tick
        elif msg.type == "note_off" or (msg.type == "note_on" and
                                        msg.velocity == 0):
            if msg.note in live:
                t0 = live.pop(msg.note)
                events.append((t0 * sec_per_tick, msg.note,
                               (abs_tick - t0) * sec_per_tick))
    total = sum(e[2] for e in events) + 0.5
    assert total > 0, "empty midi"
    y = np.zeros(int(total * SR) + SR // 2, dtype=np.float64)
    for start, midi, dur in events:
        if dur <= 0:
            continue
        f = 440.0 * 2 ** ((midi - 69) / 12)
        n = max(1, int(dur * SR))
        t = np.arange(n) / SR
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
        env = np.minimum(1, np.minimum(np.arange(n) / (0.01 * SR),
                                       (n - np.arange(n)) / (0.03 * SR)))
        env = np.clip(env, 0, 1)
        i0 = int(start * SR)
        y[i0:i0 + n] += tone * env * 0.5
    y /= max(np.abs(y).max(), 1e-6)
    assert np.abs(y).max() > 0.01, "silent wav!"
    wavfile.write(wav_path, SR, (y * 0.8 * 32767).astype(np.int16))
    return wav_path


def convert_abc_file(abc_path, out_dir):
    out_dir = Path(out_dir)
    abc = Path(abc_path).read_text(encoding="utf-8")
    notes, key = parse_abc(abc)
    assert notes, f"no notes parsed in {abc_path}"
    stem = Path(abc_path).stem
    mid = out_dir / f"{stem}.mid"
    wav = out_dir / f"{stem}.wav"
    to_midi(notes, str(mid))
    midi_to_wav(str(mid), str(wav))
    return mid, wav, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [inp] if inp.is_file() else sorted(inp.glob("*.abc"))
    ok = 0
    for f in files:
        try:
            mid, wav, key = convert_abc_file(f, out)
            print(f"OK {f.name} key={key} -> {mid.name},{wav.name}")
            ok += 1
        except Exception as e:
            print(f"FAIL {f.name}: {e}")
    print(f"{ok}/{len(files)} converted")


if __name__ == "__main__":
    main()
