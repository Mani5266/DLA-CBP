import json
r = json.load(open('outputs/results.json', encoding='utf-8'))
d = r['dataset']
print('keys:', d['keys'])
print('meters:', d['meters'])
print('mean_len:', round(d['mean_tune_len'], 1))
print('top_chars:', d['top_chars'][:6])
g = r['generation']
print('gen:', {k: v for k, v in g.items()
               if k in ('attempts', 'valid', 'valid_raw', 'valid_repaired',
                        'validity_rate', 'temperatures')})
print('per_temp:', g.get('per_temperature'))
print('orig:', g.get('originality'))
print('tabla:', g.get('tabla_extension', {}).get('status'))
for t in ['E6_u128', 'E3_d03', 'E6_u512', 'var_cnn', 'var_many2many',
          'var_stateful', 'E4_bn', 'E1_rmsprop', 'E11_b32_lr1e-3']:
    e = r['experiments'][t]
    print(t, 'params=', e['config'].get('params'), 'val=',
          {k: round(v, 3) for k, v in e['val'].items()})
print('E12:', r['experiments']['E12_cpu']['sec_per_epoch'],
      r['experiments']['E12_gpu']['sec_per_epoch'])
print('csv lines:', sum(1 for _ in open('outputs/experiments.csv')) - 1)
