"""Bounded 1,500-bid integration check; NOT the paper's experiment."""
import json
from pathlib import Path
import subprocess
import sys

output = Path(sys.argv[1])
root = output / 'data'


def run(*args):
    command = [sys.executable, *map(str, args)]
    print('RUN:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


run('-m', 'pytest', 'tests', '-q', '--junitxml', output / 'tests.xml')
run('ca_workflow.py', 'generate', '--root', root, '--items', 300, '--bids', 1500,
    '--train', 4, '--valid', 2, '--test', 2, '--seed', 2700)
run('ca_workflow.py', 'collect', '--root', root, '--workers', 2, '--time-limit', 3, '--max-solutions', 10)
run('ca_workflow.py', 'train', '--root', root, '--epochs', 2, '--batch-size', 2, '--device', 'cuda')
for solver in ['gurobi', 'scip']:
    for mode in ['baseline', 'search', 'fixing']:
        run('ca_workflow.py', 'evaluate', '--root', root, '--name', f'{solver}-{mode}',
            '--solver', solver, '--mode', mode, '--time-limit', 3)
# Check authors' supplied checkpoint on real-sized graphs, retaining upstream graph behavior.
run('ca_workflow.py', 'evaluate', '--root', root, '--name', 'provided-ca-checkpoint',
    '--checkpoint', 'models/CA.pth', '--graph-mode', 'upstream', '--time-limit', 3)
results = sorted((root / 'evaluation').glob('*/results.json'))
run('ca_workflow.py', 'summarize', '--reference', root / 'evaluation/gurobi-baseline/results.json',
    *results, '--output', output / 'summary.json')
summary = json.loads((output / 'summary.json').read_text())
assert all(row['feasible'] == 2 for row in summary['summary'])
assert all(row['mean_relative_primal_gap'] >= 0 for row in summary['summary'])
print('Full-size bounded pipeline passed; no paper-performance claim.', flush=True)
