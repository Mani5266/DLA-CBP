import re
from collections import Counter
txt = open('data/processed/train.txt', encoding='utf-8').read()
tunes = [t for t in txt.split('\n\n') if t.strip()]
print('tunes:', len(tunes))
lead_nl = sum(1 for t in tunes if t.startswith('\n'))
print('leading-newline tunes:', lead_nl)
bal = Counter()
ends = Counter()
for t in tunes:
    s = t.strip()
    o, c = s.count('|:'), s.count(':|')
    bal['equal' if o == c else ('more_open' if o > c else 'more_close')] += 1
    tail = s.replace(' ', '').replace('\n', '')[-3:]
    ends[tail] += 1
print('repeat balance:', dict(bal))
print('top endings:', ends.most_common(8))
