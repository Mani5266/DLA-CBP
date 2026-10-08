import re
from pathlib import Path
for f in ['gen_T0.6_0', 'gen_T0.8_0', 'gen_T1.0_0', 'gen_T1.2_0',
          'gen_tabla_01']:
    t = Path(f'outputs/generated/{f}.abc').read_text(encoding='utf-8')
    m = re.search(r'(?m)^M:(.*)$', t)
    k = re.search(r'(?m)^K:(.*)$', t)
    print(f, 'chars=', len(t), 'bars=', t.count('|'), 'M=',
          m.group(1).strip() if m else '?', 'K=',
          k.group(1).strip().split()[0] if k else '?')
