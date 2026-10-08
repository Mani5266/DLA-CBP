"""Record E5 (analysis) and E8/E9/E10 (reuse) entries in results.json."""
import sys
sys.path.insert(0, 'src')
from common import load_results, save_results

r = load_results()
ex = r['experiments']
g = r.get('generation', {})
mm = r['main_model']['val']

ex['E5_activation_init'] = {
    'status': 'done_as_analysis',
    'note': 'No separate training run: 8-epoch budget cannot isolate '
            'activation/init effects on the full LSTM. Delivered as written '
            'analysis in report Ch.5 (E2 clipping collapse + Char-CNN ReLU '
            'evidence + LayerNorm E4). Labeled analysis, not experiment.'}
ex['E8_temperature'] = {
    'status': 'done_reuse_phase7',
    'note': 'Phase 7 generation reused, not re-run.',
    'validity_rate': g['validity_rate'], 'attempts': g['attempts'],
    'per_temperature': g.get('per_temperature')}
ex['E9_stateful'] = {
    'status': 'done_reuse_phase5',
    'note': 'Phase 5 stateful variant reused (shuffle=False asserted, '
            'states reset each epoch).',
    'val': ex['var_stateful']['val']}
ex['E10_family'] = {
    'status': 'done_reuse_phases4_6',
    'note': 'Char-RNN (full budget) vs Char-CNN / many-to-many / small-RNN '
            '(reduced budget). Budget mismatch stated in report.',
    'char_rnn_val_acc': mm['accuracy'],
    'char_cnn_val_acc': ex['var_cnn']['val']['accuracy'],
    'many2many_val_acc': ex['var_many2many']['val']['accuracy'],
    'small_rnn_val_acc': r['baselines']['simple_rnn'].get('accuracy')}
save_results(r)
print('E5/E8/E9/E10 recorded')
