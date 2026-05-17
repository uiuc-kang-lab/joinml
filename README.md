# JoinML — Multi-Table Semantic Joins with Statistical Guarantees

`joinml` is a Python library for **analytical join queries over unstructured
data**: COUNT / SUM / AVG / MIN / MAX / MEDIAN / TopK aggregations whose join
predicate is an expensive ML model (an "Oracle" — an LLM call, a re-ID model,
a fine-tuned classifier) rather than a hash key. Given a fixed Oracle budget,
the library returns a point estimate of the aggregate plus a confidence
interval that comes with formal statistical guarantees.

The recommended algorithm is **Blocking-augmented Sampling (BaS)**, which
adaptively combines embedding-based blocking on high-similarity tuples with
importance sampling on the rest. It minimiazes RRMSE while preserving CI
coverage, and is described in the accompanying paper (Zhu et al., SIGMOD 2026;
arXiv:2603.16153).

---

## Use cases

Any analytical query of the form
```sql
SELECT AGG(g(t₁, t₂, …))
FROM   T₁ JOIN T₂ JOIN …
ON     ML_PREDICATE(t₁, t₂, …)
```
where the predicate is expensive enough that exhaustive evaluation over
`|T₁| × |T₂| × …` would cost too much.

Concretely:
- **Entity resolution analytics** — count duplicates, average size difference,
  median age gap across catalogs.
- **Plagiarism / paraphrase detection** —
  `SELECT COUNT(*) FROM article JOIN db ON NL("paraphrased?")`.
- **Multi-camera / multi-modal re-identification** — average travel time of
  vehicles across cameras, max similarity across a corpus.
- **TopK heavy hitters** — most-mentioned entity across a multimodal join.

---

## Install

```bash
git clone <repo-url>
cd join_ml
uv pip install -e .              # or:  pip install -e .
```

Python ≥ 3.10. Heavy dependencies (numpy, pandas, scipy, sentence-transformers,
numba) come from `pyproject.toml` / `requirements.txt`.

---

## Quickstart (3 steps)

### 1. Set up a dataset

A dataset directory is a folder under `data/` shaped as:
```
data/<dataset>/
├── data/table0.csv          # source records, one per row, column "join_col" is the join key
├── data/table1.csv          # second table (omit for self-joins)
└── oracle_labels/00.csv     # ground-truth matched tuples, format: "id0,id1"
```
For the datasets used in the paper, run:
```bash
bash reproduction/setup/download_data.sh paper
```

### 2. Precompute the proxy similarity tensor

The first joinml-adapt run on a (dataset, proxy) pair builds the similarity
tensor and caches it as `.npy`. For large datasets this can take hours and
tens of GB — do it once, ahead of time:
```bash
uv run python reproduction/setup/precompute_embeddings.py \
    --dataset_config configs/company-minilm.yml
```
The cache lives at `../.cache/joinml/<dataset>_<proxy>_scores.npy`
(configurable via `--cache_path` or `Config.cache_path`).

### 3. Run a query

```bash
uv run python run_joinml.py \
    --dataset_config configs/ecomm-q7.yml \
    --task joinml-adapt \
    --oracle_budget 10000 \
    --internal_loop 100 \
    --ci True
```

Output goes to `<dataset>-<task>-<proxy>-<w_exp>_<aggregator>.jsonl`, one line
per independent run, with fields `est`, `gt`, `lbs`, `ubs`, `coverages`.
Inspect:
```bash
uv run python scripts/check_results.py <output>.jsonl
uv run python scripts/check_guarantees.py <output>.jsonl
```

---

## Algorithms

`joinml-adapt` is the recommended default. Other algorithms are baselines or
specialised variants:

| `--task` | Algorithm | Notes |
|---|---|---|
| `joinml-adapt` | **Blocking-augmented Sampling (BaS)** — adaptive blocking + IS | Default. Beats baselines on most queries. |
| `joinml-fixed` | BaS with a fixed (non-adaptive) blocking ratio | Ablation reference |
| `importance` | Weighted Wander Join (IS proportional to proxy similarity) | Sampling-only baseline |
| `uniform` | Uniform sampling | Naïve baseline |
| `blocking-noci` / `blocking-ci` | Embedding-based blocking | Threshold-based baseline |
| `joinml-recall` / `joinml-precision` | BaS for selection (target-recall / target-precision) | Selection queries |
| `joinml-topk` | BaS for Top-K heavy hitters | Top-K queries |
| `large-join`, `uniform-scale`, `importance-scale`, `blocking-noci-scale` | Streaming / scaled variants | Very large cross products |
| `recall` | Exhaustive Oracle, computes recall@budget | Diagnostic |

---

## Configuration

Each dataset has a YAML config in `configs/`. Required fields:
```yaml
data_path: data                # root for CSVs and oracle labels
dataset_name: ecomm-q7         # subdir name under data_path
proxy: all-MiniLM-L6-v2        # text embedder | image proxy | string-sim | multimodal
proxy_score_cache: True        # cache proxy.npy under cache_path
aggregator: count              # count | sum | avg | min | max | median
is_self_join: True
confidence_level: 0.95
bootstrap_trials: 1000
proxy_normalizing_style: proportional   # proportional | sqrt
```

CLI flags can override anything in the YAML. The most useful tuning knobs:

| Flag | Purpose |
|---|---|
| `--oracle_budget N` | total Oracle calls allowed |
| `--max_blocking_ratio α` | cap on the fraction of budget spent on blocking (default 0.2; paper recommends 0.15–0.30) |
| `--w_exp γ` | exponent on proxy weight (default 1) — sample ∝ sim^γ |
| `--allocation_search {prefix, subset, auto, evt}` | β-search mode. `subset` enumerates all 2^K subsets when K≤12; `evt` enables the GEV-fit allocator for MIN/MAX. |
| `--defensive_mix_ratio ε` | mixes IS weights with uniform `(1−ε)W + ε/N`; bounds worst-case importance ratios |
| `--force_block_concentrated True` | when high-similarity match rate ≫ low-similarity match rate, force-block the top stratum (recommended for AVG on concentrated workloads) |
| `--var_shrinkage λ` | Bayesian shrinkage of per-stratum variances toward the pooled mean (recommend λ≈100 for noisy small pilots) |
| `--two_stage_allocation True` | top up to 30% of n_target, re-allocate, then run to final n_target |
| `--internal_loop N` | repeat the experiment N times with seeds `seed..seed+N-1` |
| `--ci True/False` | compute bootstrap-t CIs |
| `--aggregator <name>` | override the YAML's aggregator |
| `--variant_tag <s>` | partitions output under `runs/<tag>/` |

## Repository layout

```
joinml/                  # main package
├── algs/                #   per-task algorithm modules
├── proxy/               #   proxy / embedding implementations
├── plugins/             #   SUPG selection plugin
├── dataset_loader.py    #   join-dataset abstraction
├── oracle.py            #   ground-truth interface
├── estimates.py         #   result containers
├── config.py            #   Config dataclass
├── utils.py             #   sampling + CI helpers
└── run.py               #   task dispatcher
configs/                 # YAML configs per dataset
scripts/                 # generic harness (bas_run.sh, slurm_bas.sh,
                         #   check_results.py, check_guarantees.py,
                         #   eval_against_targets.py, ...)
reproduction/            # paper-reproduction scripts (see reproduction/README.md)
block_noci/              # blocking-threshold tables read by the
                         #   `blocking-noci` / `blocking-ci` baselines
data/                    # (gitignored) datasets — see data/SEMBENCH.md
                         #   for the SemBench-derived ecomm-q*/movie-q* queries
runs/                    # (gitignored) experiment outputs
results.py               # paper's published numbers (reference)
```

---

## Reproduction

See [`reproduction/README.md`](reproduction/README.md) for a step-by-step
walkthrough mapped to each section of the paper (§7.2 statistical guarantees,
§7.3 RRMSE, §7.5 ablation, §7.6 sensitivity).

---

## Citation

If you use this library, please cite:

```
@article{zhu2026accelerating,
  title={Accelerating Approximate Analytical Join Queries over Unstructured Data with Statistical Guarantees},
  author={Zhu, Yuxuan and Jin, Tengjun and Mo, Chenghao and Kang, Daniel},
  journal={Proceedings of the ACM on Management of Data},
  year={2026},
  publisher={ACM New York, NY, USA}
}
```