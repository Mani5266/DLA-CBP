import json
r = json.load(open('outputs/results.json', encoding='utf-8'))
mm = r['main_model']
print('MAIN train:', mm['train'])
print('MAIN val  :', mm['val'])
print('MAIN cfg  :', mm['config'])
print(f"{'run':22s} {'loss':>6s} {'acc':>6s} {'ppl':>6s} {'ep':>3s} {'s/ep':>5s}")
for k in sorted(r['experiments']):
    e = r['experiments'][k]
    v = e.get('val', {})
    print(f"{k:22s} {v.get('loss',0):6.3f} {v.get('accuracy',0):6.3f} "
          f"{v.get('perplexity',0):6.1f} {e.get('epochs_run',0):3d} "
          f"{e.get('sec_per_epoch',0):5.0f}")
print('baselines:', {k: v for k, v in r.get('baselines', {}).items()})
