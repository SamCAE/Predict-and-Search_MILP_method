"""Auditable CA runner. Original research scripts and GNN weights are unchanged.

See docs/CA_REPRODUCTION.md for the distinction between paper settings,
upstream behavior, and the explicitly corrected graph mode.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

import gurobipy as gp
import numpy as np
import pyscipopt as scp
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from GCN import BipartiteNodeData, GNNPolicy


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # GPU scatter reductions may still be nondeterministic; CPU is the reference.
    torch.set_num_threads(1)


def checked_entries(root, splits):
    manifest = json.loads((root / 'instances.json').read_text())
    entries = [e for e in manifest['instances'] if e['split'] in splits]
    if not entries:
        raise ValueError('No instances in the requested splits')
    for entry in entries:
        if sha256(root / entry['path']) != entry['sha256']:
            raise ValueError(f"Instance changed: {entry['path']}")
    return entries


def generate(args):
    import ecole
    if args.root.exists() and any(args.root.iterdir()):
        raise ValueError('Use a new empty --root; generation never overwrites data')
    args.root.mkdir(parents=True, exist_ok=True)
    entries = []
    index = 0
    for split, count in [('train', args.train), ('valid', args.valid), ('test', args.test)]:
        for number in range(count):
            generator = ecole.instance.CombinatorialAuctionGenerator(
                n_items=args.items, n_bids=args.bids)
            generator.seed(args.seed + index)
            instance = next(generator)
            path = Path('instance') / split / f'ca_{number:04d}.lp'
            (args.root / path).parent.mkdir(parents=True, exist_ok=True)
            instance.write_problem(str(args.root / path))
            entries.append(dict(path=str(path), split=split, seed=args.seed + index,
                                sha256=sha256(args.root / path)))
            index += 1
    save_json(args.root / 'instances.json', dict(
        generator='ecole.instance.CombinatorialAuctionGenerator',
        ecole_reported_version=ecole.__version__, n_items=args.items, n_bids=args.bids,
        other_parameters='Ecole defaults; see environment lock and Ecole API',
        provenance='New instances; author generator settings/seeds were not released',
        instances=entries))
    print(f'Generated {len(entries)} instances in {args.root}', flush=True)


def normalize(features):
    features = torch.tensor(features, dtype=torch.float32)
    low, high = features.amin(0), features.amax(0)
    return ((features - low) / (high - low).clamp_min(1e-12)).clamp(1e-5, 1)


def extract_graph(path, mode='corrected'):
    if mode == 'upstream':
        import helper
        # Upstream otherwise pickles CUDA tensors, which its loader cannot handle.
        helper.device = torch.device('cpu')
        a, mapping, variables, constraints, binary = helper.get_a_new2(str(path))
        constraints = torch.nan_to_num(constraints, nan=1., posinf=10., neginf=-10.)
        return a, mapping, variables, constraints, binary
    model = scp.Model()
    model.hideOutput()
    model.readProblem(str(path))
    variables = sorted(model.getVars(), key=lambda v: v.name)
    if any(v.vtype() != 'BINARY' for v in variables):
        raise ValueError('This runner is for binary CA instances only')
    mapping = {v.name: i for i, v in enumerate(variables)}
    rows = [(c, model.getValsLinear(c)) for c in model.getConss()]
    rows = sorted([(c, row) for c, row in rows if row], key=lambda x: (len(x[1]), str(x[0])))
    if not rows:
        raise ValueError('CA instance has no nonempty constraints')
    vf = np.zeros((len(variables), 6))
    vf[:, 3] = -np.inf
    vf[:, 4] = np.inf
    vf[:, 5] = 1
    obj = model.getObjective()
    objective = {term.vartuple[0].name: obj[term] for term in obj}
    for name, value in objective.items():
        vf[mapping[name], 0] = value
    indices = [[], []]
    cf = []
    for row_index, (constraint, coefficients) in enumerate(rows):
        rhs, lhs = model.getRhs(constraint), model.getLhs(constraint)
        sense = 2 if lhs == rhs else 0
        if model.isInfinity(rhs):
            rhs, sense = lhs, 1
        cf.append([sum(coefficients.values()) / len(coefficients), len(coefficients), rhs, sense])
        for name, value in coefficients.items():
            col = mapping[name]
            indices[0].append(row_index)
            indices[1].append(col)
            vf[col, 1] += value / len(rows)
            vf[col, 2] += 1
            vf[col, 3] = max(vf[col, 3], value)
            vf[col, 4] = min(vf[col, 4], value)
    # Keep the upstream extra objective node, but connect it to its actual row.
    cf.append([sum(objective.values()) / max(len(objective), 1), len(objective), 0, 0])
    for name, value in objective.items():
        if value:
            indices[0].append(len(rows))
            indices[1].append(mapping[name])
    vf[:, 0] = vf[:, 0].clip(0, 20000)
    vf[~np.isfinite(vf)] = 0
    a = torch.sparse_coo_tensor(indices, torch.ones(len(indices[0])), (len(rows) + 1, len(variables)))
    result = (a, mapping, normalize(vf), normalize(cf), torch.arange(len(variables)))
    model.freeProb()
    return result


def validate_solution(model, names, values, tolerance=1e-5):
    by_name = dict(zip(names, values))
    ordered = np.array([by_name[v.VarName] for v in model.getVars()])
    if not np.isfinite(ordered).all():
        return False
    for v, value in zip(model.getVars(), ordered):
        if value < v.LB - tolerance or value > v.UB + tolerance:
            return False
        if v.VType != gp.GRB.CONTINUOUS and abs(value - round(value)) > tolerance:
            return False
    activity = model.getA() @ ordered
    for c, lhs in zip(model.getConstrs(), activity):
        if c.Sense == '<' and lhs > c.RHS + tolerance:
            return False
        if c.Sense == '>' and lhs < c.RHS - tolerance:
            return False
        if c.Sense == '=' and abs(lhs - c.RHS) > tolerance:
            return False
    return True


def collect_one(job):
    root, entry, settings = job
    root = Path(root)
    out = root / 'labels' / entry['split'] / Path(entry['path']).stem
    if out.with_suffix('.json').exists():
        old = json.loads(out.with_suffix('.json').read_text())
        if old['settings'] != settings or old['instance_sha256'] != entry['sha256']:
            raise ValueError(f'Label settings changed; use a new root: {out}')
        for suffix in ('.sol', '.bg'):
            if sha256(out.with_suffix(suffix)) != old['hashes'][suffix]:
                raise ValueError(f'Changed label file: {out}{suffix}')
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(root / entry['path']), env=env) as model:
        if model.ModelSense != -1 or any(v.VType != 'B' for v in model.getVars()):
            raise ValueError('Expected a binary maximization CA model')
        model.Params.TimeLimit = settings['time_limit']
        model.Params.Threads = 1
        model.Params.Seed = settings['seed']
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = settings['max_solutions']
        model.Params.LogFile = str(out.with_suffix('.log'))
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.optimize()
        if not model.SolCount:
            raise RuntimeError(f'No feasible labels for {entry["path"]}')
        names = [v.VarName for v in model.getVars()]
        solutions, objectives = [], []
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            values = model.getAttr('Xn')
            if not validate_solution(model, names, values):
                raise RuntimeError('Solver label failed independent row/bound validation')
            solutions.append(values)
            objectives.append(model.PoolObjVal)
        data = dict(var_names=names, sols=np.asarray(solutions, dtype=np.float32),
                    objs=np.asarray(objectives, dtype=np.float64))
    graph = extract_graph(root / entry['path'], settings['graph_mode'])
    for suffix, data_to_save in [('.sol', data), ('.bg', graph)]:
        with out.with_suffix(suffix).open('wb') as f:
            pickle.dump(data_to_save, f)
    save_json(out.with_suffix('.json'), dict(settings=settings, instance_sha256=entry['sha256'],
        solution_count=len(solutions), wall_seconds=time.perf_counter() - started,
        hashes={s: sha256(out.with_suffix(s)) for s in ('.sol', '.bg')}))
    print(f'Collected {entry["path"]}: {len(solutions)} solutions', flush=True)


def collect(args):
    entries = checked_entries(args.root, ['train', 'valid'])
    settings = {k: getattr(args, k) for k in ['time_limit', 'max_solutions', 'seed', 'graph_mode']}
    jobs = [(str(args.root), e, settings) for e in entries]
    if args.workers == 1:
        for job in jobs:
            collect_one(job)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # Spawn keeps CUDA/SCIP state out of forked workers; exceptions propagate.
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            list(pool.map(collect_one, jobs))


def marginal_target(solutions, objectives, temperature):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('temperature must be positive and finite')
    weight = torch.softmax(torch.as_tensor(objectives, dtype=torch.float64) / temperature, dim=0)
    return (weight @ torch.as_tensor(solutions, dtype=torch.float64)).float()


def load_training_graphs(root, split, graph_mode, temperature, max_labels):
    result = []
    for entry in checked_entries(root, [split]):
        base = root / 'labels' / split / Path(entry['path']).stem
        meta = json.loads(base.with_suffix('.json').read_text())
        if meta['instance_sha256'] != entry['sha256'] or meta['settings']['graph_mode'] != graph_mode:
            raise ValueError('Graph mode or label provenance mismatch')
        for suffix in ['.sol', '.bg']:
            if sha256(base.with_suffix(suffix)) != meta['hashes'][suffix]:
                raise ValueError(f'Label changed: {base}{suffix}')
        with base.with_suffix('.bg').open('rb') as f:
            a, mapping, vf, cf, binaries = pickle.load(f)
        with base.with_suffix('.sol').open('rb') as f:
            sol = pickle.load(f)
        if len(binaries) != len(mapping):
            raise ValueError('Only all-binary CA models are supported')
        indices = {name: i for i, name in enumerate(sol['var_names'])}
        columns = [indices[name] for name in mapping]
        solutions = np.round(sol['sols'][:max_labels, columns])
        graph = BipartiteNodeData(cf.float(), a._indices().long(),
                                 torch.ones((a._nnz(), 1)), vf.float())
        graph.target = marginal_target(solutions, sol['objs'][:max_labels], temperature)
        graph.num_nodes = len(vf) + len(cf)
        result.append(graph)
    return result


def logits(policy, graph):
    return policy(graph.constraint_features, graph.edge_index, graph.edge_attr, graph.variable_features)


def train(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    loaders = {s: DataLoader(load_training_graphs(args.root, s, args.graph_mode,
                        args.temperature, args.max_labels), batch_size=args.batch_size,
                        shuffle=s == 'train', num_workers=0) for s in ['train', 'valid']}
    out = args.root / 'training'
    if out.exists():
        raise ValueError('training already exists; preserve it or use a new root')
    out.mkdir()
    config = {k: v for k, v in vars(args).items() if k != 'func'}
    config['root'] = str(args.root)
    config['instances_sha256'] = sha256(args.root / 'instances.json')
    save_json(out / 'config.json', config)
    model = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, stale = float('inf'), 0
    for epoch in range(args.epochs):
        record = dict(epoch=epoch)
        for split, loader in loaders.items():
            model.train(split == 'train')
            total = 0.
            with torch.set_grad_enabled(split == 'train'):
                for graph in loader:
                    graph = graph.to(device)
                    loss = F.binary_cross_entropy_with_logits(logits(model, graph), graph.target, reduction='sum')
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite training loss')
                    if split == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total += loss.item()
            record[split + '_loss'] = total / len(loader.dataset)
        if record['valid_loss'] < best:
            best, stale = record['valid_loss'], 0
            torch.save(model.state_dict(), out / 'model_best.pth')
        else:
            stale += 1
        torch.save(model.state_dict(), out / 'model_last.pth')
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(record, flush=True)
        if args.patience and stale >= args.patience:
            break


def select_partial(names, probabilities, k0, k1):
    if min(k0, k1) < 0 or k0 + k1 > len(names):
        raise ValueError(f'k0+k1={k0+k1} exceeds {len(names)} binary variables')
    order = sorted(range(len(names)), key=lambda i: (float(probabilities[i]), names[i]))
    return {**{names[i]: 0 for i in order[:k0]},
            **{names[i]: 1 for i in (order[-k1:] if k1 else [])}}


def ca_parameters(solver, mode, profile):
    if mode == 'baseline':
        return 0, 0, 0
    if profile == 'upstream':
        return 400, 0, 0 if mode == 'fixing' else 10
    if mode == 'fixing':
        return 600, 0, 0
    return (600, 0, 1) if solver == 'gurobi' else (400, 0, 10)


def solve_gurobi(path, partial, delta, args, log_path):
    curve = []
    with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(path), env=env) as model:
        model.Params.TimeLimit, model.Params.Threads = args.time_limit, 1
        model.Params.Seed, model.Params.MIPFocus = args.seed, args.mip_focus
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 0
        model.Params.LogFile = str(log_path)
        variables = {v.VarName: v for v in model.getVars()}
        if partial:
            alphas = []
            for i, (name, value) in enumerate(partial.items()):
                if args.mode == 'fixing':
                    variables[name].LB = variables[name].UB = value
                else:
                    alpha = model.addVar(lb=0., name=f'ps_alpha_{i}')
                    model.addConstr(alpha >= variables[name] - value)
                    model.addConstr(alpha >= value - variables[name])
                    alphas.append(alpha)
            if alphas:
                model.addConstr(gp.quicksum(alphas) <= delta, name='ps_neighborhood')
        def incumbent(m, where):
            if where == gp.GRB.Callback.MIPSOL:
                curve.append([m.cbGet(gp.GRB.Callback.RUNTIME), m.cbGet(gp.GRB.Callback.MIPSOL_OBJ)])
        model.optimize(incumbent)
        solution = {name: v.X for name, v in variables.items()} if model.SolCount else None
        return dict(objective=model.ObjVal if solution else None, status=int(model.Status),
                    solver_seconds=model.Runtime, incumbent_curve=curve, solution=solution)


def solve_scip(path, partial, delta, args, log_path):
    model = scp.Model()
    model.hideOutput()
    model.readProblem(str(path))
    model.setParam('limits/time', args.time_limit)
    model.setParam('parallel/maxnthreads', 1)
    for param in ['randomseedshift', 'lpseed', 'permutationseed']:
        model.setParam('randomization/' + param, args.seed)
    model.setHeuristics(scp.SCIP_PARAMSETTING.AGGRESSIVE)
    model.setLogfile(str(log_path))
    variables = {v.name: v for v in model.getVars()}
    alphas = []
    for i, (name, value) in enumerate(partial.items()):
        if args.mode == 'fixing':
            model.fixVar(variables[name], value)
        else:
            alpha = model.addVar(name=f'ps_alpha_{i}', vtype='C', lb=0)
            model.addCons(alpha >= variables[name] - value)
            model.addCons(alpha >= value - variables[name])
            alphas.append(alpha)
    if alphas:
        model.addCons(scp.quicksum(alphas) <= delta)
    curve = []
    class Incumbent(scp.Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(scp.SCIP_EVENTTYPE.BESTSOLFOUND, self)
        def eventexit(self):
            self.model.dropEvent(scp.SCIP_EVENTTYPE.BESTSOLFOUND, self)
        def eventexec(self, event):
            curve.append([self.model.getSolvingTime(), self.model.getSolObjVal(self.model.getBestSol())])
    model.includeEventhdlr(Incumbent(), 'incumbents', 'Record primal incumbent history')
    model.optimize()
    solution = {name: model.getVal(v) for name, v in variables.items()} if model.getNSols() else None
    result = dict(objective=model.getObjVal() if solution else None, status=str(model.getStatus()),
                  solver_seconds=model.getSolvingTime(), incumbent_curve=curve, solution=solution)
    model.freeProb()
    return result


def evaluate(args):
    seed_all(args.seed)
    entries = checked_entries(args.root, ['test'])
    if args.limit:
        entries = entries[:args.limit]
    defaults = ca_parameters(args.solver, args.mode, args.profile)
    k0, k1, delta = [default if getattr(args, key) is None else getattr(args, key)
                     for key, default in zip(['k0', 'k1', 'delta'], defaults)]
    if delta < 0:
        raise ValueError('delta must be nonnegative')
    policy = None
    if args.mode != 'baseline':
        checkpoint = args.checkpoint or args.root / 'training' / 'model_best.pth'
        config_path = checkpoint.parent / 'config.json'
        if config_path.exists():
            trained_mode = json.loads(config_path.read_text())['graph_mode']
            if trained_mode != args.graph_mode:
                raise ValueError('Checkpoint and evaluation graph modes disagree')
        elif args.graph_mode != 'upstream':
            raise ValueError('Checkpoint without metadata requires --graph-mode upstream')
        policy = GNNPolicy().to(args.device)
        policy.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        policy.eval()
    out = args.root / 'evaluation' / args.name
    if out.exists():
        raise ValueError('Evaluation name already exists; choose a new --name')
    out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}
    config.update(k0=k0, k1=k1, delta=delta, threads=1,
                  checkpoint_sha256=sha256(checkpoint) if policy else None)
    save_json(out / 'config.json', config)
    results = []
    for entry in entries:
        path = args.root / entry['path']
        started = time.perf_counter()
        partial = {}
        if policy:
            a, mapping, vf, cf, binary = extract_graph(path, args.graph_mode)
            graph = BipartiteNodeData(cf, a._indices(), torch.ones((a._nnz(), 1)), vf).to(args.device)
            with torch.no_grad():
                probabilities = logits(policy, graph).sigmoid().cpu().tolist()
            partial = select_partial(list(mapping), probabilities, k0, k1)
        preprocess = time.perf_counter() - started
        solver = solve_gurobi if args.solver == 'gurobi' else solve_scip
        result = solver(path, partial, delta, args, out / (path.stem + '.log'))
        result.update(instance=entry['path'], instance_sha256=entry['sha256'],
                      preprocessing_seconds=preprocess, wall_seconds=time.perf_counter() - started,
                      selected_variables=len(partial))
        if result['solution']:
            solution = result['solution']
            with gp.Env(params={'OutputFlag': 0}) as env, gp.read(str(path), env=env) as original:
                if original.ModelSense != -1:
                    raise ValueError('Expected CA maximization')
                if not validate_solution(original, list(solution), list(solution.values())):
                    raise RuntimeError('Incumbent is infeasible in ORIGINAL problem')
                original_obj = original.ObjCon + sum(v.Obj * solution[v.VarName] for v in original.getVars())
                if not math.isclose(original_obj, result['objective'], rel_tol=1e-7, abs_tol=1e-5):
                    raise RuntimeError('Objective mismatch')
            distance = sum(abs(solution[name] - value) for name, value in partial.items())
            if distance > delta + 1e-5:
                raise RuntimeError('Incumbent violates neighborhood')
            result.update(original_feasible=True, hamming_distance=distance)
        else:
            result.update(original_feasible=False, hamming_distance=None)
        save_json(out / (path.stem + '.json'), result)
        results.append(result)
        save_json(out / 'results.json', results)
        print({k: v for k, v in result.items() if k not in ['solution', 'incumbent_curve']}, flush=True)


def summarize(args):
    paths = list(dict.fromkeys([args.reference] + args.results))
    groups = [json.loads(path.read_text()) for path in paths]
    keyed = [{row['instance']: row for row in group} for group in groups]
    names = set(keyed[0])
    if any(set(group) != names for group in keyed):
        raise ValueError('All result files must cover exactly the same test instances')
    for name in names:
        if len({group[name]['instance_sha256'] for group in keyed}) != 1:
            raise ValueError('Cannot compare results from different instances')
    if any(keyed[0][name]['objective'] is None for name in names):
        raise ValueError('Reference must contain a feasible objective for every instance')
    bks = {name: max(group[name]['objective'] for group in keyed if group[name]['objective'] is not None)
           for name in names}
    output = []
    for path, group in zip(paths, keyed):
        feasible = [name for name in sorted(names) if group[name]['objective'] is not None]
        complete = len(feasible) == len(names)
        absolute = [abs(group[name]['objective'] - bks[name]) for name in feasible]
        relative = [abs(group[name]['objective'] - bks[name]) / (abs(bks[name]) + 1e-10) for name in feasible]
        output.append(dict(result=str(path), instances=len(names), feasible=len(feasible),
            mean_objective=float(np.mean([group[n]['objective'] for n in feasible])) if complete else None,
            mean_absolute_primal_gap=float(np.mean(absolute)) if complete else None,
            mean_relative_primal_gap=float(np.mean(relative)) if complete else None))
    save_json(args.output, dict(reference=str(args.reference),
        BKS_source='Reference incumbent updated by better incumbents in ALL supplied runs (CA maximization)',
        bks=bks, summary=output))
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, function in [('generate', generate), ('collect', collect), ('train', train), ('evaluate', evaluate)]:
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, required=True)
        sub.add_argument('--seed', type=int, default=0)
        sub.set_defaults(func=function)
        if name != 'generate':
            sub.add_argument('--graph-mode', choices=['corrected', 'upstream'], default='corrected')
        if name in ['train', 'evaluate']:
            sub.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        if name == 'generate':
            sub.add_argument('--items', type=int, required=True, help='Not specified in the paper; choose explicitly')
            sub.add_argument('--bids', type=int, default=1500)
            for split, count in [('train', 240), ('valid', 60), ('test', 100)]:
                sub.add_argument('--' + split, type=int, default=count)
        elif name == 'collect':
            sub.add_argument('--workers', type=int, default=1)
            sub.add_argument('--time-limit', type=float, default=3600)
            sub.add_argument('--max-solutions', type=int, default=500)
        elif name == 'train':
            sub.add_argument('--lr', type=float, default=.003)
            sub.add_argument('--batch-size', type=int, default=8)
            sub.add_argument('--epochs', type=int, default=9999)
            sub.add_argument('--patience', type=int, default=0, help='0 disables early stopping; not specified by paper')
            sub.add_argument('--temperature', type=float, default=1000, help='Upstream CA energy temperature; paper formula implies 1')
            sub.add_argument('--max-labels', type=int, default=50)
        elif name == 'evaluate':
            sub.add_argument('--name', required=True)
            sub.add_argument('--solver', choices=['gurobi', 'scip'], default='gurobi')
            sub.add_argument('--mode', choices=['baseline', 'search', 'fixing'], default='search')
            sub.add_argument('--profile', choices=['paper', 'upstream'], default='paper')
            sub.add_argument('--time-limit', type=float, default=1000)
            sub.add_argument('--mip-focus', type=int, default=1, help='Gurobi: 1 for experiments; 0 for default BKS reference')
            sub.add_argument('--checkpoint', type=Path)
            sub.add_argument('--limit', type=int)
            for parameter in ['k0', 'k1', 'delta']:
                sub.add_argument('--' + parameter, type=int)
    sub = commands.add_parser('summarize')
    sub.add_argument('--reference', type=Path, required=True)
    sub.add_argument('results', nargs='+', type=Path)
    sub.add_argument('--output', type=Path, required=True)
    sub.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
