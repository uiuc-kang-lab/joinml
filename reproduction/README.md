# Reproducing BaS paper results

This directory mirrors the structure of the paper (Zhu et al., SIGMOD 2026 /
arXiv:2603.16153) so each experiment can be reproduced independently. Every
subfolder corresponds to a section of the preprint, with a per-section
`run.sh` (or themed scripts) and a short `README.md` documenting the inputs,
outputs, and expected numbers.

```
reproduction/
├── setup/                          # one-time data + embedding preparation
├── 7.2_statistical_guarantees/     # §7.2 — CI coverage (Figure 5)
├── 7.3_rrmse_main/                 # §7.3 — Main RRMSE comparison (Figure 7)
├── 7.5_allocation_ablation/        # §7.5 — Adaptive vs fixed-α allocation
├── 7.6_sensitivity/                # §7.6 — Knob sensitivities (Figures 13a/b)
└── README.md (this file)
```

---

## Step 0 — Install

From the repo root:
```bash
uv pip install -e .            # or:  pip install -e .
```

## Step 1 — Data setup ([`setup/`](setup/))

Datasets and the cached similarity tensors are *not* shipped with the
repository. Two preparation steps are required before any experiment will run.

### Download the raw datasets

```bash
bash reproduction/setup/download_data.sh paper       # every dataset used in the paper
# or one at a time:
bash reproduction/setup/download_data.sh quora company
```

The script wraps `data/download.sh` (a Google-Drive-backed downloader for
the public datasets) and skips any dataset already present under `data/`.
SemBench-derived datasets (`ecomm-q*`, `movie-q*`) are shipped under
[`../SemBench/`](../SemBench) and don't need a download.

After this step the layout should be:
```
data/<dataset>/data/table0.csv          # and table1.csv for two-table joins
data/<dataset>/oracle_labels/00.csv     # ground-truth match tuples
```

### Precompute the proxy similarity tensors

```bash
# All datasets used in the paper (Quora ~28 GB, Flickr30k ~38 GB on disk!):
bash reproduction/setup/precompute_all.sh

# Or just one (e.g. Company with the all-MiniLM-L6-v2 encoder):
uv run python reproduction/setup/precompute_embeddings.py \
    --dataset_config configs/company-minilm.yml
```
Cached `.npy` tensors land under `../.cache/joinml/`. The cache is keyed by
`(dataset_name, proxy)` so re-runs of the same experiment skip recomputation.
Override location with `CACHE_PATH=/path/to/cache bash ...precompute_all.sh`.

### Smoke check

Verify everything is wired up with a fast end-to-end test:
```bash
bash reproduction/setup/verify_setup.sh
```
This launches one short (~5 min) `joinml-adapt` run per dataset to confirm
data + embedding caches + algorithm are all healthy.

---

## §7.2 — Statistical guarantees (Figure 5)

Goal: confirm that BaS's confidence intervals achieve the nominal 95%
coverage across budgets and aggregators.

The experiment runs `joinml-adapt` with `--ci True` and `--internal_loop=500`
on four (dataset, aggregator) combinations: Roxford-COUNT, Quora-AVG,
Webmasters-SUM, Ecomm-Q9-MEDIAN. Metric: 95th-percentile of `|est − gt| / (CI
half-width)` across the 500 runs — must be ≤ 1 for the CI to be valid.

Run:
```bash
bash reproduction/7.2_statistical_guarantees/run.sh
```

Evaluate:
```bash
for f in runs/sec7.2/*.jsonl; do
    uv run python scripts/check_guarantees.py "$f"
done
```
Acceptance: every cell prints `95th percentile error ratio ≤ 1`.

## §7.3 — Main RRMSE comparison (Figure 7)

Goal: reproduce the per-(dataset, budget) RRMSE comparison against the
baselines in Figure 7 of the paper.

The Figure-7 sweep covers 16 (dataset, aggregator) combinations across the
budget grid in the paper. We split it into two scripts because the
non-linear aggregators (MIN/MAX/MEDIAN) need a different allocator flag.

```bash
# Linear aggregators (COUNT, SUM, AVG)
bash reproduction/7.3_rrmse_main/linear_sweep.sh

# Non-linear aggregators (MIN, MAX, MEDIAN) — uses --allocation_search evt
bash reproduction/7.3_rrmse_main/nonlinear_sweep.sh
```

Both scripts iterate cells sequentially via `scripts/bas_run.sh`. They write
each cell's RRMSE / CI-validity / vs-target lines to stdout as they go, and
each cell's raw jsonl into `runs/sec7.3-linear/` or `runs/sec7.3-nonlinear/`.

If you'd rather distribute the cells across a slurm or other parallel
launcher, wrap `bash scripts/bas_run.sh ...` in your scheduler's submission
command and run the `jobs=()` rows in parallel.

Evaluate the full sweep:
```bash
uv run python scripts/eval_against_targets.py --grid runs/sec7.3-*/*.jsonl
```

## §7.5 — Adaptive-allocation ablation (Figure 10)

Goal: show that BaS's adaptive allocation is close to the
oracle-best allocation and well above the worst-allocation baseline.

The paper sweeps a fixed blocking ratio over {10%, 20%, 30%, 40%, 50%} on six
datasets (Quora, Company, Roxford, Flickr30K, Webmasters, VeRi) and compares
each fixed-α run to BaS-adaptive. The fixed-α variant is the existing
`joinml-fixed` task; the adaptive variant is `joinml-adapt`.

```bash
bash reproduction/7.5_allocation_ablation/run.sh
```
Runs 36 cells (6 datasets × 5 fixed-α + 1 adaptive) sequentially. Outputs
land under `runs/sec7.5-fixed-<α>/` and `runs/sec7.5-adapt/`. The paper's
"Optimal" allocation is whichever α minimises RRMSE on each dataset; "Worst"
is whichever maximises it.

## §7.6 — Sensitivity analysis (Figures 11–13)

Three sub-experiments map onto the existing knobs:

| Figure | What varies | Script |
|---|---|---|
| 13a | `--max_blocking_ratio` ∈ {0.10, 0.15, 0.20, 0.25, 0.30} on Webmasters / Flickr30k | `7.6_sensitivity/alpha_ratio.sh` |
| 13b | `--w_exp` ∈ {0.5, 1, 1.5, 2, 3, 5} on Company | `7.6_sensitivity/weight_exponent.sh` |
| (extra) | MIN/MAX proposal ablations: defmix, force-block, GEV-fit, larger-α | `7.6_sensitivity/min_max_proposals.sh` |

`ablation_full.sh` and `ablation_focused.sh` in this directory run the
broader Phase-C variant matrix we used during reproduction (force-block,
allocation-search-subset, strata-size variants) on the outlier cells.

```bash
bash reproduction/7.6_sensitivity/alpha_ratio.sh
bash reproduction/7.6_sensitivity/weight_exponent.sh
bash reproduction/7.6_sensitivity/min_max_proposals.sh
```
All four sensitivity scripts run sequentially.

---

## Evaluation helpers

`scripts/check_results.py FILE`     — per-budget RRMSE  
`scripts/check_guarantees.py FILE`  — per-budget 95-th-percentile CI error ratio  
`scripts/eval_against_targets.py [--grid] FILES …` — cross-file comparison
against the paper's published BaS targets, with optional dataset×budget grid view.

`scripts/check_topk_results.py`, `check_selection.py`, `check_synthetic_results.py`
cover the corresponding non-aggregation paper experiments.
