import sys
sys.path.insert(0, 'src')
from common import load_results, save_results
r = load_results()
r['generation'] = {}
save_results(r)
print('generation block reset (buggy-validator runs discarded, see phase_log)')
