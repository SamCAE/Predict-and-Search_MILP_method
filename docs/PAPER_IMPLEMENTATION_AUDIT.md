# Paper versus released implementation

**Verdict: the central predict-and-search idea is present, but this checkout is not a
verified faithful reproduction of the paper. There are confirmed implementation bugs,
material parameter differences, and missing experimental provenance.** Passing the added
bounded tests establishes functionality for those tested cases, not correctness of every
upstream code path or reproduction of the reported performance.

Source: user-provided Han et al., *A GNN-Guided Predict-and-Search Framework for
Mixed-Integer Linear Programming*, ICLR 2023, arXiv:2302.05636v4, 6 March 2023.
The supplied PDF SHA-256 is
`52397e5a34a02e5dc6102633705890feb165237f5d883c545fa8dd6d8b013ad9`.
Relevant text was extracted from the supplied file and Tables 2-4 and Section 4/Algorithm 1
were visually checked. Paper instructions were treated as source material, not commands.
Audit date: 2026-09-18. Repository commit: `02f342af44ba17bfaa92b5c4db0bcbe90fb94151`.
The initial working tree was clean. Original research scripts/checkpoints remain unchanged.

## Confirmed defects

| Finding | Code evidence | Consequence / bounded verification |
|---|---|---|
| All three evaluation scripts fail to parse | `PredictAndSearch_GRB.py:68`, `PredictAndSearch_SCIP.py:67`, `FixingStrategy_SCIP.py:67`: over-indented IP branch | `ast.parse` raises `IndentationError` before any solve. An environment alone cannot fix this. |
| Objective edges attach to the wrong node | `helper.py:398` adds objective edges to row 0; constraint edges start at row 0 at line 447; objective features are appended last at line 458 | First constraint gets unrelated/duplicate edges; objective node has no incoming edges. The 3-variable fixture produces `[[2,2,1],[0,1,1],[0,0,0]]` instead of `[[1,1,0],[0,1,1],[1,1,1]]`. Sparse duplicate entries really contribute repeated messages; they are not just a display issue. |
| Constraint normalization divides by zero | `helper.py:481-486`: no guard for constant feature ranges | The all-`<=` fixture yields 3 NaNs in the sense column. `GCN.py:204` replaces NaNs with 1 afterwards, making the pipeline dependent on undocumented repairs. |
| CUDA-generated training pickles are not portable to the provided loader | `helper.py` puts tensors on global CUDA device; `gurobi.py` pickles them; `GCN.py:208` reconstructs with CPU `torch.FloatTensor` | The constructor fails on CUDA input in the prepared environment. A CPU-only machine also cannot normally unpickle CUDA tensors without mapping. New collection stores CPU tensors. |
| Evaluation paths disagree with README | Evaluation reads `./instance/CA`, while README says `instance/test/CA` | Generated instances placed per README are not found. All scripts default to IS, training defaults to WA, and evaluation assumes at least 100 files. |
| GPU hardcoded when loading weights | Evaluation calls `torch.load(... map_location='cuda:0')` even though DEVICE can be CPU | CPU fallback fails before prediction. The new runner loads onto the requested device. |
| Evaluation logs can be overwritten | Fixing and search SCIP scripts use the same `..._Predect&Search` directory and filenames | Later fixing runs can replace earlier search evidence. New runs have explicit distinct names and refuse overwrite. |
| Unsafe collection defaults/error reporting | `gurobi.py` loops over IP/WA/IS/CA/NNV, starts 100 workers, skips solely on BG existence, never checks worker exit codes | Missing directories block the README command; incomplete/corrupted pairs can be reused; worker failures need not make the parent fail. New collector is CA-only, checks hashes, and propagates worker exceptions. |

Machine-readable evidence: [upstream-audit.json](validation/upstream-audit.json).
The added tests reproduce the adjacency/NaN issues and verify the repaired graph independently
on a hand-computable case. They intentionally do not mark the original scripts as fixed.

## Material paper/code differences

| Aspect | Supplied paper | Released implementation | Assessment |
|---|---|---|---|
| CA PS+Gurobi | Table 4: `k0=600, k1=0, delta=1` | `PredictAndSearch_GRB.py`: `400,0,10` | Definite, consequential mismatch: different selected set and neighborhood. |
| CA PS+SCIP | Table 4: `400,0,10` | `400,0,10` | Matches. |
| CA Fixing+SCIP | Table 4: `600,0,0` | `FixingStrategy_SCIP.py`: fixes 400 zeros | Definite mismatch. Appendix D/Figure 5 labels the comparison Fixing+Gurobi, while Table 4 and the released fixing script say SCIP; experimental provenance is ambiguous. |
| Training batch/LR | Section 4: batch 8, Adam start LR 0.003 | `trainPredictModel.py:28-30`: batch 4, LR 0.001 | Definite mismatch. |
| Energy weighting | Equations (3)-(4): `exp(-c^T x)` for minimization | CA uses `exp(objective/1000)` via `EnergyWeightNorm('CA')=-1000` | Maximization sign is correct, but the 1,000 temperature is not disclosed in the paper. It changes marginal targets. |
| Numerics of weights | Probability normalization | Raw float32 `exp` then divide by sum | Can overflow: objective 100,000 at CA scale produces infinity. This is a demonstrated possible failure, not a claim that the published dataset hits it. New code uses stable softmax. |
| Number of solutions | Appendix B does not state a pool cap, pool-search mode, or label truncation | Collect up to 500 with `PoolSearchMode=2`, then `GraphDataset` takes only first 50 | Important undocumented implementation choices. |
| Train/valid/test counts | 240/60/100 | Loader splits an arbitrary number of unsorted filenames 80/20; no test dataset supplied | Ratios match only if exactly 300 development instances are provided. Reproducible IDs/splits are absent. |
| CA instance generation | Ecole, as in Gasse et al.; Table 3 gives maximum size only | No generator, generation parameters, seeds, or actual instances | Exact dataset reproduction unavailable. 1,500 variables alone does not determine the distribution. |
| Constraint/variable features | Table 2 lists coefficient edge features, integer indicator, and 12 positional features | CA has six variable features, no positional features; 12 positional features used only for IP; helper marks binary rather than all integer variables | Positional restriction is not stated in the supplied feature table. Integer-vs-binary difference does not affect all-binary CA. |
| Edge coefficients | Table 2 describes actual constraint coefficients | Helper stores all ones; loader/inference overwrite edges with ones; `LayerNorm(1)` maps every scalar edge input to the same learned bias | Edge coefficient information cannot reach messages. For ordinary CA rows all coefficients are 1, so that particular loss is less material than for general MILPs; the graph-row defect still matters for CA. |
| GNN layers | Section 4 says one embedding perceptron and two half-convolutions | Two Linear/ReLU embedding stages and four half-convolutions (two full variable/constraint rounds) | Text/code discrepancy. Appendix A has a generic iterative formulation, so the intended depth needs author confirmation rather than declaring the architecture mathematically invalid. |
| Training reproducibility | Epoch count and seeds unspecified | 9,999 epochs; Python seed only; unsorted file list; torch initialization/shuffle unseeded; cuDNN benchmark enabled | Neither exact checkpoint training nor run-to-run identity is established. |
| Evaluation metric | BKS from 3,600s single-thread Gurobi, updated by better tested incumbents; per-instance absolute/relative primal gap | Only solver logs; no BKS runner, aggregate metrics, or figure-generation script | Solver MIP gap cannot substitute for paper primal gap. The new runner supplies BKS comparison and incumbent histories. |
| Software/hardware | Gurobi 9.5.2, SCIP 8.0.1, PyTorch 1.10.2; Xeon Gold 5117/V100 hardware | Historical YAML, now evaluated here using Gurobi 13.0.3/SCIP 10.0.3/PyTorch 2.6/RTX 4070 | A modern compatibility run cannot substantiate the paper's wall-time gains. |

There are additional non-CA parameter mismatches: PS+Gurobi uses the SCIP-style radii for
IP (1 instead of 10), IS (15 instead of 20), and WA (5 instead of 10). WA selects 600 ones
instead of the paper's 500. These were identified by inspection, not benchmarked here.

Table 3 reports CA as 6,396 constraints / 1,500 variables and IS as 600 / 1,500. These
counts look unusual for the standard generators and may reflect a labeling mistake,
different generator parameters, or preprocessing. This audit **does not assume a swap**.
The supplied PDF visually confirms those printed entries. Ask for original LPs and the
generator script to resolve the issue.

## Components that are consistent

- The network outputs one sigmoid probability per variable. Selecting the smallest `k0`
  and largest `k1` binary probabilities implements the partial-solution rule.
- For CA maximization, assigning larger weight to larger objective values is appropriate.
  The sign of the CA energy is not a bug; the undisclosed temperature is the mismatch.
- The original auxiliary variables are continuous, whereas Algorithm 1 calls them binary.
  This is **not a feasible-set error**: for binary `x` and binary target `x*`, constraints
  `alpha >= x-x*`, `alpha >= x*-x`, `alpha >= 0`, and `sum(alpha)<=delta` project exactly to
  `sum(abs(x-x*))<=delta`. The new tests compare both solvers to exhaustive enumeration.
- One neighborhood solve is consistent with Section 3.2.2; repeated neighborhood expansion
  or a fallback unrestricted solve is not required by the stated algorithm.
- Validation-best checkpoint selection, 64-dimensional embeddings, Adam, 1,000-second
  solver caps, and Gurobi's primal emphasis are broadly consistent with the description.

The paper's Proposition 1 compares **optimal subproblem values** with the same partial
solution. It does not guarantee that a time-limited neighborhood solve beats fixing,
that the neighborhood contains an original optimum, or that PS always beats the base solver.
The source is sufficient for this restricted interpretation; bounded tests supply no
additional universal performance guarantee.

## What was added and what remains unresolved

The added `ca_workflow.py` imports the original GNN and provides deterministic explicit
splits, CPU graph serialization, a graph mode with corrected row indexing/normalization,
stable weighted BCE, paper/upstream CA parameter profiles, CPU/GPU checkpoint loading,
both solver backends, original-feasibility checks, separate logs, and BKS aggregation.
The corrected mode still keeps the upstream architecture, objective node, unit edges,
and objective clipping. It is a useful controlled repair, not a complete reimplementation
of every literal statement in the paper. Retrain when changing graph conventions.

The original `models/CA.pth` loads, but no metadata proves which data, graph convention,
loss temperature, or training settings produced it. For conservative compatibility it is
evaluated using the upstream graph mode. Structural loadability is not authentication of
the checkpoint's experimental provenance.

To claim exact reproduction, obtain from the authors: original generated instances or
full generator configuration/version/seeds; train/validation/test identifiers; checkpoint
training seed and stopping rule; intended graph construction/edge features; energy
temperature and solution-selection policy; and the actual Gurobi/fixing parameter sets
and original BKS/trajectory logs. Re-run historical solver versions on comparable hardware
for timing comparisons. The missing information cannot be recovered from this checkout.

Use [CA_REPRODUCTION.md](CA_REPRODUCTION.md) for the complete tested procedure and bounded
test scope. Only CA and the explicitly listed fixtures were executed; IP/WA/IS were not.
