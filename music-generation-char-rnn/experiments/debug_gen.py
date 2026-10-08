import random, torch
import sys
sys.path.insert(0, 'src')
from common import PROCD, load_vocab
from generate import rebuild, validate

chars, c2i, V = load_vocab()
vocab_set = set(chars)
i2c = {i: c for c, i in c2i.items()}
ckpt = torch.load('outputs/checkpoints/main_full.pt', map_location='cpu')
model = rebuild(ckpt['config'], V)
model.load_state_dict(ckpt['state'])
model.eval()

train_tunes = [t for t in (PROCD / 'train.txt').read_text(
    encoding='utf-8').split('\n\n') if t.strip()]
rng = random.Random(42)
for trial in range(3):
    src = rng.choice(train_tunes)
    cut = rng.randint(40, min(110, len(src) - 20))
    seed_txt = src[:cut]
    cur = [c2i[c] for c in seed_txt]
    with torch.no_grad():
        while len(cur) < 800:
            x = torch.tensor([cur[-64:]], dtype=torch.long)
            nxt = torch.multinomial(torch.softmax(
                model(x).squeeze(0) / 0.8, -1), 1).item()
            cur.append(nxt)
            if len(cur) > 120 and i2c[cur[-2]] == '|' and \
                    i2c[cur[-1]] == ']':
                break
    tune = ''.join(i2c[i] for i in cur)
    ok, reasons = validate(tune, vocab_set)
    print(f'--- trial {trial} ok={ok} len={len(tune)} reasons={reasons}')
    print('HEAD:', repr(tune[:120]))
    print('TAIL:', repr(tune[-120:]))
    print('counts |::', tune.count('|:'), ':|:', tune.count(':|'),
          'bars:', tune.count('|'))
