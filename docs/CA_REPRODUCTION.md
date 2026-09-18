# Running CA in WSL

The prepared environment is `predict-search-ca` in **Ubuntu-24.04**, at
`/home/ubuntu24/anaconda3/envs/predict-search-ca`. Your existing Gurobi license works.
CUDA is available on the RTX 4070 Laptop GPU.

**This checkout does not contain enough information to reproduce the exact paper numbers.**
The original data, CA generator parameters/seeds, training stopping rule, and checkpoint
provenance are missing. The original evaluation scripts also do not parse. The commands
below give a complete, executable experiment with explicit choices, using the original
`GCN.GNNPolicy`. Read [the audit](PAPER_IMPLEMENTATION_AUDIT.md) before interpreting results.

## 1. Activate the prepared environment

In PowerShell, enter the correct distribution:

```powershell
wsl -d Ubuntu-24.04
```

Then, in WSL bash:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate predict-search-ca
cd /mnt/c/Users/samer/Desktop/Github/Neural_Diving/Predict-and-Search_MILP_method
python -m pip check
python -m pytest tests -q
```

Installed versions: Python 3.12.14, PyTorch 2.6.0+cu124, PyG 2.6.1, NumPy 1.26.4,
SciPy 1.13.1, Gurobi/gurobipy 13.0.3, SCIP 10.0.3, PySCIPOpt 6.2.1. The conda Ecole
package is labeled 0.8.2 but its Python version metadata reports 0.8.1. Both are recorded.
These are a tested modern compatibility environment, **not** the paper's software versions.

For a fresh installation on this WSL distribution, run `bash scripts/setup_wsl.sh`.
For the exact resolved packages in a separate environment:

```bash
conda create -y -n predict-search-ca-copy --file docs/validation/conda-linux-64-explicit.txt
conda activate predict-search-ca-copy
python -m pip install -r requirements-wsl-lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip check
```

`pip-freeze.txt` is an inspection record, not an installation requirements file: conda
packages in it have build-machine `file://` paths. Use `requirements-wsl-lock.txt` instead.
No Gurobi license files were copied or changed. The original `py38.yaml` remains intact;
it pins an old CUDA/Python stack, omits Ecole, and embeds an author's absolute prefix.

## 2. Generate 240 training, 60 validation, and 100 test instances

Use a new experiment directory. Keeping generated data on WSL's Linux filesystem usually
reduces I/O overhead; source code may remain in this checkout.

```bash
export CA_RUN="$HOME/predict-search-runs/ca-300items-1500bids-seed0-corrected"
python ca_workflow.py generate \
  --root "$CA_RUN" --items 300 --bids 1500 \
  --train 240 --valid 60 --test 100 --seed 0
```

**300 items is an explicit experimental choice, not a recovered author setting.**
1,500 bids is consistent with the paper's reported variable count, but its Table 3 reports
6,396 CA constraints, which this choice does not reproduce. Do not label this generated
dataset as the authors' dataset. Replace `--items` and other generator settings if the
authors supply them. All other parameters use the pinned Ecole defaults documented in
the [Ecole generator API](https://doc.ecole.ai/py/en/stable/reference/instances.html#combinatorial-auction).

Each instance gets its own seed: 0-239 for training, 240-299 for validation, 300-399 for
testing. The manifest `instances.json` records split, seed, path, and SHA-256. Generation
refuses to overwrite a nonempty directory. Training never consumes the test split.

## 3. Collect training and validation solutions and graphs

```bash
python ca_workflow.py collect \
  --root "$CA_RUN" --workers 4 \
  --time-limit 3600 --max-solutions 500 --seed 0 --graph-mode corrected
```

Each Gurobi solve uses one thread, `PoolSearchMode=2`, and up to 500 feasible solutions,
matching the repository's collection settings. The 3,600-second limit matches Appendix B.
The paper does not specify the pool-search mode or pool size. Four worker processes are
a practical local choice; the upstream default of 100 workers is unsuitable here.

Labels are written under `labels/train` and `labels/valid`. Each instance has `.sol`, `.bg`,
`.json`, and solver `.log` files. Every stored solution is checked against original rows,
variable bounds, and integrality. Graphs are stored on CPU so they can be used by either
CPU or GPU training. Re-running collection skips completed, checksum-matching labels;
changed settings or corrupted labels cause a failure instead of silently reusing them.

`corrected` fixes objective-edge indexing and constant-column normalization while keeping
the repository's six variable features, four constraint features, extra objective node,
unit edge features, and GNN architecture. This is a documented repaired variant, not a
claim of exact paper equivalence. To study upstream behavior, use a **separate run root**
and pass `--graph-mode upstream` to collection, training, and evaluation consistently.

## 4. Train

```bash
python ca_workflow.py train \
  --root "$CA_RUN" --device cuda --seed 0 --graph-mode corrected \
  --batch-size 8 --lr 0.003 --epochs 9999 --patience 0 \
  --temperature 1000 --max-labels 50
```

Batch size 8 and learning rate 0.003 match paper Section 4. The epoch cap of 9,999 comes
from the repository; the paper does not give the training duration/stopping rule.
`--patience 0` disables early stopping. If you prefer an explicitly different practical
training budget, choose e.g. `--epochs 1000 --patience 100` and report that choice.

For CA maximization, label weights are `softmax(objective / temperature)`. Temperature
1,000 and the first 50 pool solutions match the repository. Equations (3)-(4), read
literally after converting maximization to minimization, instead imply temperature 1;
the paper does not disclose the temperature or truncation. Use `--temperature 1` to test
that interpretation, and `--max-labels 500` to use all collected solutions. These choices
can materially change the labels. Do not tune them using the test set.

The stable marginal-target BCE used here is mathematically equivalent to a weighted sum
of per-solution BCE at the same temperature/solutions; a test verifies this identity and
its gradient. Numerically it uses logits, instead of the upstream sigmoid/log calculation
with added `1e-8` guards, so behavior at saturated probabilities can differ.
The loss is summed over variables/graphs for each optimizer step, as upstream does.

Outputs:

- `training/model_best.pth`: lowest validation-loss checkpoint, used for evaluation.
- `training/model_last.pth`: latest checkpoint.
- `training/history.jsonl`: per-epoch training and validation losses.
- `training/config.json`: seeds, graph mode, hyperparameters, instance-manifest hash.

Training refuses to overwrite an existing training directory. There is no optimizer-state
resume implementation. Use a new root for a new experiment, or deliberately preserve/move
the old training directory before restarting. CPU is supported with `--device cpu`.
GPU scatter reductions and time-limited MILP solves are not guaranteed bitwise reproducible.

## 5. Evaluate the 100 held-out instances

The commands below run sequentially and use one solver thread. The solver limit excludes
graph extraction/inference, matching the structure of the original scripts; both solver
time and additional preprocessing/wall time are saved. Paper timing does not clearly
specify whether preprocessing was included.

```bash
# Long-run reference for the paper's BKS-based metric.
python ca_workflow.py evaluate --root "$CA_RUN" --name bks \
  --solver gurobi --mode baseline --time-limit 3600 --mip-focus 0

# Gurobi baseline and predict-and-search: k0=600, k1=0, delta=1.
python ca_workflow.py evaluate --root "$CA_RUN" --name gurobi \
  --solver gurobi --mode baseline --time-limit 1000
python ca_workflow.py evaluate --root "$CA_RUN" --name ps-gurobi \
  --solver gurobi --mode search --profile paper --time-limit 1000

# SCIP baseline and predict-and-search: k0=400, k1=0, delta=10.
python ca_workflow.py evaluate --root "$CA_RUN" --name scip \
  --solver scip --mode baseline --time-limit 1000
python ca_workflow.py evaluate --root "$CA_RUN" --name ps-scip \
  --solver scip --mode search --profile paper --time-limit 1000

# Optional fixing baseline from Table 4: k0=600, k1=0, delta=0.
python ca_workflow.py evaluate --root "$CA_RUN" --name fixing-scip \
  --solver scip --mode fixing --profile paper --time-limit 1000
```

Gurobi experiments use `MIPFocus=1`. The long-run BKS command uses default focus 0 because
the reference sentence specifies single-thread Gurobi but does not specify its emphasis.
SCIP uses aggressive primal heuristics, as the repository does; this is distinct from
setting SCIP's entire `FEASIBILITY` emphasis preset. Each evaluation name must be new.

Each `evaluation/NAME/results.json` contains objective, solver status, times, incumbent
curve, selected-variable count, Hamming distance, original-feasibility check, and the
solution. The status `optimal` for a neighborhood problem only means **optimal inside
that neighborhood**, not optimal in the original MILP.

## 6. Compute the paper's metric

```bash
python ca_workflow.py summarize \
  --reference "$CA_RUN/evaluation/bks/results.json" \
  "$CA_RUN/evaluation/gurobi/results.json" \
  "$CA_RUN/evaluation/ps-gurobi/results.json" \
  "$CA_RUN/evaluation/scip/results.json" \
  "$CA_RUN/evaluation/ps-scip/results.json" \
  "$CA_RUN/evaluation/fixing-scip/results.json" \
  --output "$CA_RUN/summary.json"
```

Omit the fixing result if that optional run was skipped. For each test instance, BKS is
the maximum feasible objective among the reference and **all supplied methods**. The
summary averages `abs(OBJ-BKS)` and `abs(OBJ-BKS)/(abs(BKS)+1e-10)` over instances.
It does not use the solver-reported MIP gap or average objective values before taking ratios.
Missing incumbents produce null full-set means, not a deceptively improved feasible-only mean.
Instance sets and checksums must match across runs.

For a Figure 2 style curve, take the best recorded incumbent up to each solver-time point
for each instance, compare it to this same final BKS, then average per-instance relative
gaps. A time point with missing incumbents must be disclosed, not silently omitted.
Gain in Table 1 is `(baseline_mean_abs_gap - PS_mean_abs_gap)/baseline_mean_abs_gap * 100`;
it is undefined when the baseline gap is zero.

Upper-bound compute budgets are substantial: labeling alone is 300 solver-hours
(roughly 75 hours with four workers if every solve reaches its limit), BKS is 100 hours,
and each 100-instance/1,000-second method is about 27.8 hours if run serially. Actual solves
can finish early. These full experiments were **not** launched during setup.

## Supplied checkpoint and upstream settings

After generation, you can test the repository's `models/CA.pth` without collecting labels
or training. It loads successfully in the new environment. Preserve its original graph
convention; the runner rejects using an unannotated checkpoint with corrected graphs.

```bash
python ca_workflow.py evaluate --root "$CA_RUN" --name supplied-ca-upstream \
  --checkpoint models/CA.pth --graph-mode upstream \
  --solver gurobi --mode search --profile upstream --time-limit 1000
```

`--profile upstream` uses `(400,0,10)` for CA, including Gurobi, to match the released code.
Changing this to `--profile paper` uses Gurobi's documented `(600,0,1)`. For a training run
closer to the released configuration, use upstream graphs, batch size 4, learning rate
0.001, temperature 1,000, 50 labels, and 9,999 epochs. That still cannot recover unknown
training seeds or the original checkpoint's data history.

## Bounded tests already performed / repeat them

```bash
python -m pytest tests -q
# Complete small pipeline, with 20 items, 80 bids, and 4/2/2 split:
bash scripts/smoke_ca.sh runs/my-new-smoke
```

A separate 300-item, 1,500-bid pipeline is recorded under
`docs/validation/full-size/`. It uses 4/2/2 instances, two GPU training epochs,
3-second solver limits, and at most 10 labels per instance. This checks execution and
feasibility, not the paper's performance claims. Its input hashes, software versions,
bounded-run logs, unit-test XML, and summary are retained with a computation manifest.

The [validation results](VALIDATION_RESULTS.md) describe the completed checks. Generated
LPs, labels, logs, and checkpoints under the recorded `data/` directory remain available
locally but are ignored by Git; the summary, test report, run logs, and manifest are retained.

The original research files were kept unchanged. Use `ca_workflow.py` for the supported
commands above; running the old top-level evaluation scripts still reproduces their
documented syntax errors. This keeps the audit attributable to the exact upstream commit.
