"""Record bounded, repeatable evidence about the unmodified upstream code."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import helper
from GCN import GNNPolicy, GraphDataset


report = {'scope': 'Upstream commit 02f342af44ba17bfaa92b5c4db0bcbe90fb94151; original files unmodified',
          'python': sys.version, 'packages': {}, 'syntax': {}}
for package in ['torch', 'torch-geometric', 'numpy', 'scipy', 'gurobipy', 'pyscipopt', 'ecole']:
    report['packages'][package] = importlib.metadata.version(package)
for filename in ['GCN.py', 'helper.py', 'gurobi.py', 'trainPredictModel.py',
                 'PredictAndSearch_GRB.py', 'PredictAndSearch_SCIP.py', 'FixingStrategy_SCIP.py']:
    try:
        ast.parse(Path(filename).read_text())
        report['syntax'][filename] = 'OK'
    except SyntaxError as exc:
        report['syntax'][filename] = {'error': type(exc).__name__, 'line': exc.lineno, 'message': exc.msg}
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'tiny.lp'
    path.write_text('Maximize\n obj: 4 x1 + 3 x2 + 2 x3\nSubject To\n'
                    ' row_a: x1 + x2 <= 1\n row_b: x2 + x3 <= 1\n'
                    'Binary\n x1 x2 x3\nEnd\n')
    helper.device = torch.device('cpu')
    a, mapping, vf, cf, _ = helper.get_a_new2(str(path))
    report['graph'] = {'variable_order': list(mapping), 'actual_dense_adjacency': a.to_dense().tolist(),
        'expected_dense_adjacency': [[1, 1, 0], [0, 1, 1], [1, 1, 1]],
        'nonfinite_constraint_features': int((~torch.isfinite(cf)).sum())}
    policy = GNNPolicy()
    report['edge_layer_norm_outputs'] = policy.edge_embedding(torch.tensor([[1.], [2.], [-3.]])).tolist()
    report['raw_exp_overflows_at_ca_objective_100000'] = not bool(torch.isfinite(torch.exp(torch.tensor(100000.) / 1000.)))
    if torch.cuda.is_available():
        report['cuda_gpu'] = torch.cuda.get_device_name()
        try:
            torch.FloatTensor(cf.cuda())
            report['upstream_cuda_tensor_constructor'] = 'OK'
        except Exception as exc:
            report['upstream_cuda_tensor_constructor'] = str(exc)
report['checkpoint_sha256'] = hashlib.sha256(Path('models/CA.pth').read_bytes()).hexdigest()
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
