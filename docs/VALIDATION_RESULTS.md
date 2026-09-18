# Validation performed on 2026-09-18

**The added CA workflow passed its bounded checks. The original released evaluation
scripts remain broken, and the paper's numerical results have not been reproduced.**

Environment: Ubuntu-24.04 under WSL2, Intel Core i9-14900HX, RTX 4070 Laptop GPU (8 GB),
about 31 GiB WSL RAM. Conda environment: `/home/ubuntu24/anaconda3/envs/predict-search-ca`.
The existing Gurobi academic license successfully solved instances with 1,500 binaries and
the added neighborhood auxiliaries. The license was neither copied nor modified.

| Check | Observed result |
|---|---|
| Python/solver/GNN imports and `pip check` | Pass; no broken requirements |
| CUDA forward/backward | Pass on RTX 4070 Laptop GPU |
| Unit/invariant tests | **22 passed**, no skips, 4.17 s in the recorded run |
| Both solvers vs exhaustive enumeration | Agree on a 3-variable model for baseline, fixing, and radius-0/radius-1 neighborhoods |
| Graph construction | Corrected graph matches hand-computed incidence; test independently reproduces upstream objective-node and NaN defects |
| Loss and batching | Stable marginal BCE matches weighted BCE and gradients; batched GNN outputs match separate graph outputs |
| Generator | Same Ecole seed reproduces the same LP bytes in this environment |
| Small pipeline | 20 items/80 bids; 4 training, 2 validation, 2 test instances; two GPU epochs; 12 feasible evaluation results across both solvers and three methods |
| Full-size bounded pipeline | 300 items/1,500 bids; 4/2/2 instances; six solution pools of 10; two GPU epochs; all seven evaluation variants finish |
| Full-size returned solutions | **14/14 pass** original-row, bound, integrality, objective, and neighborhood checks |
| Supplied `models/CA.pth` | Loads with PyTorch 2.6 and produces feasible search solutions on two generated 1,500-bid instances using upstream graphs |
| Syntax of added Python and bash scripts | Pass |
| Original evaluation script syntax | Three `IndentationError`s, retained and recorded in `upstream-audit.json` |
| Computation manifest | Valid evidence record; input/output hashes verified after execution |

The full-size recorded run took **117.207 seconds**, including its unit tests and process
startup. Each optimization was limited to three seconds and one solver thread; collection
used two workers. The manifest runner enforced a 600-second overall timeout and a 1 MiB
combined-log cap. Numerical-library thread environment variables were capped at one;
these cooperative caps are not a memory bound or a strict process-tree CPU allocation.

Full-size corrected-model training losses were 1011.69 then 784.24; validation losses
were 857.83 then 559.28. These are two-epoch smoke-test losses, not a converged model.
For the two test instances, mean objective values at this short limit were:

| Method | Mean objective (maximize) |
|---|---:|
| Gurobi baseline | 23037.99 |
| Gurobi search, newly trained two-epoch model | 19348.93 |
| SCIP baseline | 22505.40 |
| SCIP search, newly trained two-epoch model | 21052.75 |
| Gurobi search, provided CA checkpoint, upstream graphs | 23041.40 |

The newly trained smoke model is worse than the baselines, as can happen with an
undertrained model and a restrictive neighborhood. This does not measure the paper's
training protocol or establish that its reported gains are true or false. The supplied
checkpoint result is also too small and too time-limited for a performance conclusion.
The smoke-run BKS is only the best incumbent among these short runs; it is **not** a
3,600-second paper reference or a certified optimum.

Evidence:

- [Full-size summary](validation/full-size/summary.json)
- [Test XML](validation/full-size/tests.xml)
- [Run manifest](validation/full-size/manifest.json)
- [Recorded commands and output](validation/full-size/stdout.txt)
- [Original-code defect evidence](validation/upstream-audit.json)
- [Conda package lock](validation/conda-linux-64-explicit.txt)
- [Complete package inventory](validation/conda-packages.json)

Regenerate the bounded full-size pipeline, from the repository root in the activated
environment, using a fresh output directory:

```bash
python scripts/validate_full_size.py runs/my-full-size-validation
```

The recorded provenance wrapper is the installed Mathbox computation-audit runner; its
exact invocation, input hashes, software, timeout, start time, and return code are in the
manifest. The script above reproduces the underlying checks without that wrapper.

No complete 240/60/100 training/evaluation experiment was launched. IP, WA, IS, historical
solver versions, and a fully coefficient-aware replacement architecture were not tested.
