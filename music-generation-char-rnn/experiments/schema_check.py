"""Phase 8 gate: results.json schema check (PRD 10.3) + fill gpu_cpu_timing."""
import sys
sys.path.insert(0, 'src')
from common import load_results, save_results

r = load_results()
errs = []
for k in ['project', 'seed', 'environment', 'dataset', 'main_model',
          'baselines', 'generation', 'experiments', 'gpu_cpu_timing']:
    if k not in r:
        errs.append(f'missing top key {k}')
mm = r['main_model']
for k in ['config', 'train', 'val']:
    if k not in mm:
        errs.append(f'main_model missing {k}')
for k in ['loss', 'accuracy', 'perplexity']:
    if k not in mm.get('val', {}):
        errs.append(f'main val missing {k}')
g = r['generation']
for k in ['attempts', 'valid', 'validity_rate', 'temperatures']:
    if k not in g:
        errs.append(f'generation missing {k}')
missing = [f"E{i}" for i in
           ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
           if not any(q.startswith(f'E{i}_') or q == f'E{i}'
                     for q in r['experiments'])]
if missing:
    errs.append(f'experiments missing: {missing}')
for k, v in r['experiments'].items():
    if isinstance(v, dict) and 'status' not in v:
        errs.append(f'{k} has no status')

cpu = r['experiments']['E12_cpu']['sec_per_epoch']
gpu = r['experiments']['E12_gpu']['sec_per_epoch']
r['gpu_cpu_timing'] = {'local_cpu_sec_per_epoch': cpu,
                       'local_gpu_sec_per_epoch': gpu,
                       'speedup': cpu / gpu,
                       'config_used': 'subset=0.05 epochs=2 batch=64 stride=2 '
                                      'seq_len=64 (identical both)'}
save_results(r)
print('schema', 'FAIL: ' + '; '.join(errs) if errs else 'PASS')
print(f"E12 {cpu:.1f}s CPU vs {gpu:.1f}s GPU = {cpu/gpu:.1f}x")
