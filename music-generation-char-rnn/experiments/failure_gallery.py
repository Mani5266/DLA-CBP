"""Failure gallery (PRD 9.3): undertrained-checkpoint sample (smoke, 5 epochs)
at T=1.0 - the "gibberish if undertrained" case from PRD 11.
NOT counted in validity stats; saved under report/ only."""
import sys
sys.path.insert(0, 'src')
import torch
from common import PROCD, ROOT, load_vocab
from generate import rebuild, validate

chars, c2i, V = load_vocab()
i2c = {i: c for c, i in c2i.items()}
ckpt = torch.load('outputs/checkpoints/smoke.pt', map_location='cpu')
model = rebuild(ckpt['config'], V)
model.load_state_dict(ckpt['state'])
model.eval()
torch.manual_seed(7)

tunes = [t.strip() for t in (PROCD / 'train.txt').read_text(
    encoding='utf-8').split('\n\n') if t.strip()]
src = tunes[3]
cur = [c2i[c] for c in src[:90]]
with torch.no_grad():
    while len(cur) < 800:
        x = torch.tensor([cur[-64:]], dtype=torch.long)
        cur.append(torch.multinomial(torch.softmax(
            model(x).squeeze(0) / 1.0, -1), 1).item())
        if len(cur) > 120 and i2c[cur[-2]] == ':' and i2c[cur[-1]] == '|':
            break
tune = ''.join(i2c[i] for i in cur)
ok, reasons = validate(tune, set(chars))
(ROOT / 'report' / 'failure_gallery.abc').write_text(tune, encoding='utf-8')
(ROOT / 'report' / 'failure_gallery.txt').write_text(
    f"smoke checkpoint (5 epochs, 15% data), temperature=1.0, seed=train tune #3\nvalid={ok}\n"
    f"reasons={reasons}\nchars={len(tune)}\n", encoding='utf-8')
print('gallery:', 'valid' if ok else f'invalid {reasons}', 'len=', len(tune))
print(repr(tune[:200]))
print(repr(tune[-200:]))
