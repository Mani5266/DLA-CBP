"""Phase 7.6 tabla extension: rhythm-focused seed through the abc model.
Honest scope (PRD 3.1): extension DEMONSTRATION, not a tabla model - the abc
representation has no timbre/bols, only pitch + duration on one line."""
import sys
sys.path.insert(0, 'src')
import torch
from common import PROCD, OUT, load_vocab, load_results, save_results
from generate import rebuild, validate, repair
from convert import convert_abc_file

SEED = ("X:999\nT:Tabla-style rhythmic sketch\nM:6/8\nL:1/16\nR:jig\nK:D\n"
        "d2d d2d ddd d2d|eee e2e eee e2e|ddd d2d AAA A2A|DDD D2D ddd d2d|")

chars, c2i, V = load_vocab()
i2c = {i: c for c, i in c2i.items()}
ckpt = torch.load('outputs/checkpoints/main_full.pt', map_location='cpu')
model = rebuild(ckpt['config'], V)
model.load_state_dict(ckpt['state'])
model.eval()

cur = [c2i[c] for c in SEED]
with torch.no_grad():
    while len(cur) < 800:
        x = torch.tensor([cur[-64:]], dtype=torch.long)
        nxt = torch.multinomial(torch.softmax(
            model(x).squeeze(0) / 1.0, -1), 1).item()
        cur.append(nxt)
        if len(cur) > 200 and i2c[cur[-2]] == ':' and i2c[cur[-1]] == '|':
            break
tune = ''.join(i2c[i] for i in cur)
ok, reasons = validate(tune, set(chars))
status = 'raw' if ok else 'invalid'
if not ok:
    tune, notes = repair(tune, SEED[:SEED.find('K:') + 4])
    ok2, _ = validate(tune, set(chars))
    status = ('repaired:' + '+'.join(notes)) if ok2 else 'invalid'
print('tabla demo:', status, 'len=', len(tune), reasons if not ok else '')
if status != 'invalid':
    p = OUT / 'generated' / 'gen_tabla_01.abc'
    p.write_text(tune, encoding='utf-8')
    mid, wav, key = convert_abc_file(p, OUT / 'generated')
    print('saved', p.name, mid.name, wav.name)
    r = load_results()
    g = r.get('generation', {})
    g['tabla_extension'] = {
        'file': 'gen_tabla_01.abc', 'status': status,
        'seed_rhythm': '6/8 jig, L:1/16 semiquaver ostinato',
        'note': 'Extension demo only: abc encodes pitch+duration, no tabla '
                'timbre/bols; model continues the driving rhythm in D major '
                'folk idiom. Not a tabla model.'}
    save_results(r)
