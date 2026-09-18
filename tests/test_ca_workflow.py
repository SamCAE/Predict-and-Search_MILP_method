import itertools
from pathlib import Path
from types import SimpleNamespace

import gurobipy as gp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ca_workflow import (ca_parameters, extract_graph, logits, marginal_target,
                         select_partial, solve_gurobi, solve_scip, validate_solution)
from GCN import BipartiteNodeData, GNNPolicy


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / 'tiny.lp'
    path.write_text('Maximize\n obj: 4 x1 + 3 x2 + 2 x3\nSubject To\n'
                    ' row_a: x1 + x2 <= 1\n row_b: x2 + x3 <= 1\n'
                    'Binary\n x1 x2 x3\nEnd\n')
    return path


def as_graph(data):
    a, mapping, vf, cf, binary = data
    graph = BipartiteNodeData(cf, a._indices(), torch.ones((a._nnz(), 1)), vf)
    graph.num_nodes = len(vf) + len(cf)
    return graph


def test_corrected_graph_matches_hand_computed_incidence(tiny):
    a, mapping, vf, cf, _ = extract_graph(tiny)
    assert list(mapping) == ['x1', 'x2', 'x3']
    assert torch.equal(a.to_dense(), torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 1., 1.]]))
    assert torch.isfinite(vf).all() and torch.isfinite(cf).all()
    assert (cf[:, 3] == 1e-5).all()  # constant column normalized safely


def test_upstream_objective_row_bug_is_reproduced(tiny):
    import helper
    helper.device = torch.device('cpu')
    a, _, _, cf, _ = helper.get_a_new2(str(tiny))
    assert a.to_dense()[-1].sum() == 0  # objective node is isolated
    assert a.to_dense()[0].sum() == 5   # objective edges attached to first constraint
    assert torch.isnan(cf[:, 3]).all()  # zero range in sense feature


def test_stable_marginals_prefer_larger_ca_objective():
    sols = [[0, 1], [1, 0]]
    result = marginal_target(sols, [999999., 1000000.], 1)
    assert torch.isfinite(result).all()
    assert result[0] > result[1]
    assert torch.allclose(result, marginal_target(sols, [-1., 0.], 1))


def test_marginal_bce_matches_solution_weighted_loss_and_gradient():
    solutions = torch.tensor([[0., 1., 1.], [1., 0., 1.]])
    objectives = torch.tensor([5., 8.])
    z = torch.tensor([.2, -.4, .7], requires_grad=True)
    weight = torch.softmax(objectives / 1000., 0)
    direct = (F.binary_cross_entropy_with_logits(z.expand_as(solutions), solutions,
                                               reduction='none').sum(1) * weight).sum()
    compressed = F.binary_cross_entropy_with_logits(z, marginal_target(solutions, objectives, 1000), reduction='sum')
    assert torch.allclose(direct, compressed)
    assert torch.allclose(torch.autograd.grad(direct, z)[0], torch.autograd.grad(compressed, z)[0])


def test_pyg_batch_matches_separate_graph_inference(tiny):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    policy = GNNPolicy().eval()
    graph = as_graph(extract_graph(tiny))
    batch = next(iter(DataLoader([graph, graph], batch_size=2)))
    with torch.no_grad():
        expected = logits(policy, graph)
        actual = logits(policy, batch)
    assert torch.allclose(actual, torch.cat([expected, expected]), atol=1e-6)


def test_selected_sets_are_disjoint_and_zero_k1_is_empty():
    assert select_partial(['a', 'b', 'c'], [.1, .7, .9], 1, 0) == {'a': 0}
    assert select_partial(['a', 'b', 'c'], [.1, .7, .9], 1, 1) == {'a': 0, 'c': 1}
    with pytest.raises(ValueError):
        select_partial(['a', 'b'], [.1, .9], 2, 1)


@pytest.mark.parametrize('solver,mode,expected', [
    ('gurobi', 'search', (600, 0, 1)), ('scip', 'search', (400, 0, 10)),
    ('scip', 'fixing', (600, 0, 0)), ('gurobi', 'baseline', (0, 0, 0))])
def test_paper_ca_settings(solver, mode, expected):
    assert ca_parameters(solver, mode, 'paper') == expected


@pytest.mark.parametrize('solver', [solve_gurobi, solve_scip])
@pytest.mark.parametrize('mode,delta', [('search', 1), ('search', 0), ('fixing', 0), ('baseline', 0)])
def test_solver_matches_exhaustive_enumeration(tiny, tmp_path, solver, mode, delta):
    partial = {} if mode == 'baseline' else {'x1': 0, 'x3': 0}
    candidates = [x for x in itertools.product([0, 1], repeat=3)
                  if x[0] + x[1] <= 1 and x[1] + x[2] <= 1
                  and (not partial or x[0] + x[2] <= delta)]
    expected = max(sum(c * v for c, v in zip([4, 3, 2], x)) for x in candidates)
    args = SimpleNamespace(time_limit=5., seed=0, mip_focus=1, mode=mode)
    result = solver(tiny, partial, delta, args, tmp_path / 'solve.log')
    assert result['objective'] == pytest.approx(expected)
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(tiny), env=env) as model:
        assert validate_solution(model, list(result['solution']), list(result['solution'].values()))


def test_original_feasibility_check_rejects_bad_solution(tiny):
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(tiny), env=env) as model:
        assert not validate_solution(model, ['x1', 'x2', 'x3'], [1, 1, 0])
        assert not validate_solution(model, ['x1', 'x2', 'x3'], [.5, 0, 0])
        assert validate_solution(model, ['x1', 'x2', 'x3'], [1, 0, 1])


def test_supplied_ca_checkpoint_loads_and_predicts(tiny):
    policy = GNNPolicy()
    policy.load_state_dict(torch.load(Path(__file__).parents[1] / 'models/CA.pth',
                                     map_location='cpu', weights_only=True))
    with torch.no_grad():
        result = logits(policy, as_graph(extract_graph(tiny, 'upstream'))).sigmoid()
    assert result.shape == (3,) and torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_forward_and_backward(tiny):
    policy = GNNPolicy().cuda()
    graph = as_graph(extract_graph(tiny)).to('cuda')
    result = logits(policy, graph)
    result.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_ecole_seed_repeats_identical_problem(tmp_path):
    import ecole
    paths = []
    for number in range(2):
        generator = ecole.instance.CombinatorialAuctionGenerator(n_items=20, n_bids=80)
        generator.seed(2026)
        path = tmp_path / f'{number}.lp'
        next(generator).write_problem(str(path))
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()
